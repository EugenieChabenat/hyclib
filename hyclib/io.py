import pathlib
import os
import pickle
import json
import logging
import collections
import contextlib

import numpy as np
import torch
from scipy import io as sio
import pandas as pd
import h5py
import mat73
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import itertools, exceptions

logger = logging.getLogger(__name__)

ILLEGAL_KEYS = ['CLASS', 'PYTABLES_FORMAT_VERSION', 'TITLE', 'VERSION']

def loadmat(filename):
    try:
        data = sio.loadmat(filename, simplify_cells=True)
    except NotImplementedError as err:
        data = mat73.loadmat(filename)
    return data

class ItemProcessed(Exception):
    pass

class Counter:
    def __init__(self):
        self.count = 0

    def update(self, n=1):
        self.count += n

def desc_hdf(filename, progress=True):
    d = {}
    
    with hdf5_reader(filename) as (reader, count):
        for k, v in tqdm(reader, total=count, disable=(not progress)):
            if isinstance(v, PData):
                v_ = v.select()
                n_rows = len(v_)
                if isinstance(v_, pd.DataFrame):
                    columns = v_.columns.tolist()
                    v = f'pd.DataFrame: shape ({n_rows}, {len(columns)}), columns {columns}'
                else:
                    v = f'pd.Series: length {n_rows}'
            elif isinstance(v, h5py.Dataset):
                v = f'h5py.Dataset: shape {v.shape}, dtype {v.dtype}'
            else:
                v = str(v)
                ks = k.split('/')
                k = '/'.join(ks[:-1] + ['attrs'] + ks[-1:])
            itertools.dict_set(d, k, v, delimiter='/')
            
    return d

def it_to_hdf(filename, data, groupname='', callback=None, errors='raise', delimiter='/', progress=True):
    assert isinstance(data, collections.abc.Iterable)
    
    total = len(data) if isinstance(data, collections.abc.Sized) else None
    groupname = pathlib.Path('/') / groupname
    
    if callback is None:
        callback = lambda k, v, logger: None
        
    if errors == 'raise':
        def error_handler(err, k, v, kind):
            raise RuntimeError(f"Error when creating {kind} with key {k}") from err
    elif errors == 'log':
        def error_handler(err, k, v, kind):
            logger.error(f"Error when creating {kind} with key {k}: {str(err)}")
    else:
        raise ValueError(f"errors must be 'raise' or 'log', but {errors} provided.")
    
    with logging_redirect_tqdm():
        for k, v in tqdm(data, total=total, disable=(not progress)):
            if k in ILLEGAL_KEYS:
                raise ValueError(f'key cannot be one of {illegal_keys} since it conflicts with pandas metadata, but {k=} provided')
                
            ks = k.split(delimiter)
            k = '/'.join(ks)
            
            try:
                new_v = callback(k, v, logger)
            except ItemProcessed:
                continue
            else:
                v = v if new_v is None else new_v
            
            print("here")
                
            if (is_tensor:=isinstance(v, torch.Tensor)) or isinstance(v, np.ndarray):
                v = v.detach().cpu().numpy() if is_tensor else v
                
                with h5py.File(filename, 'a') as f:
                    group = f.require_group(str(groupname))
                    try:
                        group.create_dataset(k, data=v)
                    except Exception as err:
                        error_handler(err, k, v, 'dataset')
                    
            elif isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                with pd.HDFStore(filename, 'a') as f:
                    try:
                        f[str(groupname / k)] = v
                    except Exception as err:
                        error_handler(err, k, v, 'pandas object')
                
            else:
                with h5py.File(filename, 'a') as f:
                    group = f.require_group(str(groupname))
                    subgroup = group.require_group('/'.join(ks[:-1])) if len(ks) > 1 else group

                    try:
                        subgroup.attrs[ks[-1]] = v
                    except Exception as err:
                        error_handler(err, k, v, 'attribute')
                        
def to_hdf(filename, data, **kwargs):
    if 'delimiter' in kwargs:
        raise TypeError("'delimiter' is not a valid keyword argument/")
        
    data = list(itertools.dict_iter(data, delimiter='/'))
    it_to_hdf(filename, data, **kwargs, delimiter='/')
                        
class PData:
    def __init__(self, hdfstore, key):
        assert isinstance(hdfstore, pd.HDFStore)
        self.hdfstore = hdfstore
        self.key = key
        
    def select(self, *args, **kwargs):
        return self.hdfstore.select(self.key, *args, **kwargs)
                        
def hdf5_generator(hdfstore, h5pyfile, groupname='', counter=None):
    """
    Not a true generator in the sense that it iterates through
    the whole file structure twice. However, this is due to the
    limitation of the h5py library.
    """
    groupname = pathlib.Path('/') / groupname
    
    pd_paths = []
    for k in hdfstore.keys():
        k = pathlib.Path(k)
        if k.is_relative_to(groupname):
            path = k.relative_to(groupname)
            pd_paths.append(path)
            if counter is not None:
                counter.update()
    
    group = h5pyfile[str(groupname)]
    hdf_paths = [pathlib.Path('')]
    def func(name, obj):
        path = pathlib.Path(name)
        if not any(path.is_relative_to(p) for p in pd_paths):
            hdf_paths.append(path)
            if counter is not None:
                if isinstance(obj, h5py.Group):
                    counter.update(len([k for k in obj.attrs.keys() if k not in ILLEGAL_KEYS]))
                else:
                    counter.update()
    group.visititems(func)
    
    def generator():
        for path in pd_paths:
            yield str(path), PData(hdfstore, str(groupname / path))

        for path in hdf_paths:
            obj = group[str(path)]
            if isinstance(obj, h5py.Group):
                for attr_name, attr in obj.attrs.items():
                    if attr_name not in ILLEGAL_KEYS:
                        yield str(path / attr_name), attr
            else:
                yield str(path), obj
            
    return generator()
    
@contextlib.contextmanager
def hdf5_reader(filename, **kwargs):
    hdfstore = pd.HDFStore(filename, 'r')
    h5pyfile = h5py.File(filename, 'r')
    counter = Counter()
    try:
        yield hdf5_generator(hdfstore, h5pyfile, counter=counter, **kwargs), counter.count
    finally:
        hdfstore.close()
        h5pyfile.close()
        
def load_hdf5_obj(obj):
    if isinstance(obj, PData):
        return obj.select()
    if isinstance(obj, h5py.Dataset):
        return obj[...]
    return obj

def from_hdf(filename, callback=None, progress=True, **kwargs):
    d = {}
    
    if callback is None:
        callback = lambda d, k, v, logger: None
        
    with hdf5_reader(filename, **kwargs) as (reader, count):
        for k, v in tqdm(reader, total=count, disable=(not progress)):
            try:
                out = callback(d, k, v, logger)
            except ItemProcessed:
                pass
            else:
                k, v = (k, v) if out is None else out
                v = load_hdf5_obj(v)
                itertools.dict_set(d, k, v, delimiter='/')
    
    return d
                
            
# def from_hdf(filename, groupname='', callback=None, progress=True):
#     """
#     Recursive iterator of an hdf5 file
    
#     callback is a function with signature (rel_path, abs_path, obj, logger) -> None or (k, v)
#         - obj can be PData, h5py.Dataset, or other primitive types.
#         - If v is pd.HDFStore, d[k] = v[abs_path]. If v is h5py.Dataset, d[k] = v[...]. Otherwise, d[k] = v.
#         - To override this behavior, implement your custom behavior and then raise ItemProcessed exception.
#     """
#     groupname = pathlib.Path('/') / groupname
    
#     if callback is None:
#         callback = lambda d, k, v, logger: None
        
#     def set_item(d, k, v):
#         try:
#             out = callback(d, k, v, logger)
#         except ItemProcessed:
#             pass
#         else:
#             k, v = (k, v) if out is None else out
#             v = load_hdf5_obj(v)
#             itertools.dict_set(d, k, v, delimiter='/')

#     d = {}
    
#     pd_paths = []
#     with pd.HDFStore(filename, 'r') as f, logging_redirect_tqdm():
#         for k in tqdm(f.keys(), disable=(not progress)):
#             k = pathlib.Path(k)
#             if k.is_relative_to(groupname):
#                 path = k.relative_to(groupname)
#                 pd_paths.append(path)
#                 set_item(d, str(path), PData(hdfstore, str(k)))
        
#     with h5py.File(filename, 'r') as f, logging_redirect_tqdm():
#         group = f[str(groupname)]
        
#         counter = Counter() # could also be done using nonlocal
#         group.visit(counter)
        
#         with tqdm(total=counter.count+1, disable=(not progress)) as pbar:
#             def func(path, obj):
#                 path = pathlib.Path(path)
#                 if any(path.is_relative_to(p) for p in pd_paths):
#                     pass
#                 elif isinstance(obj, h5py.Group):
#                     for attr_name, attr in obj.attrs.items():
#                         set_item(d, str(path / attr_name), attr)
#                 else:
#                     set_item(d, str(path), obj) # obj is a dataset
                    
#                 pbar.update()
                
#             func('', group)
#             group.visititems(func)
    
#     return d

def save(path, data, extension=None, depth=-1, overwrite=False):
    """
    Saves data at path.
    If path has a suffix and extension is provided, then 
    the suffix must match the extension, and the data 
    is stored as a single file with the specified extension at path.
    If path has a suffix and no extension is provided, then
    the extension is inferred and the data is also stored
    as a single file.
    The depth parameter is ignored in both these case.
    If path does not have suffix, then path is the directory in which
    the data is stored. The data must be a dictionary.
    The depth parameter controls the depth of the directory.
    If depth=0, then path will become a depth 0 directory, and
    if depth=-1, then path will be a directory as deep as the data dictionary.
    """
    path = pathlib.Path(path)
    
    if extension is None:
        extension = path.suffix[1:]
    
    if extension == 'pkl':
        def save_func(filename, dat):
            with open(filename, 'wb') as f:
                pickle.dump(dat, f)
    elif extension == 'json':
        def save_func(filename, dat):
            with open(filename, 'w') as f:
                json.dump(dat, f, indent=4)
    else:
        raise NotImplementedError(f"Extension {extension} is not yet implemented")
    
    if path.suffix != '': # path is a filename
        suffix = path.suffix[1:]
        if suffix != extension:
            raise ValueError(f"path suffix must match extension if suffix is present. suffix: {suffix}, extension: {extension}.")
        if not overwrite and path.is_file():
            raise exceptions.PathAlreadyExists(f"The file {str(path)} already exists.")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_func(path, data)
        
    else: # path is a directory
        for key, val in itertools.flatten_dict(data, depth=depth).items():
            filename = path / f"{'/'.join(key.split('.'))}.{extension}"
            filename.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite and filename.is_file():
                raise exceptions.PathAlreadyExists(f"The file {str(filename)} already exists.")
            save_func(filename, val)
        
def save_data(path, data_dict, **kwargs):
    save(path, data_dict, 'pkl', **kwargs)
    
def save_config(path, config, **kwargs):
    save(path, config, 'json', **kwargs)
    
def load(path, extension=None):
    path = pathlib.Path(path)
    if not path.exists():
        raise exceptions.PathNotFound(f"The path {path} does not exist.")
        
    if extension is None:
        extension = path.suffix[1:]
        
    if extension == 'pkl':
        def load_func(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    elif extension == 'json':
        def load_func(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    else:
        raise NotImplementedError(f"Extension {extension} is not yet implemented")
        
    if path.is_file():
        data = load_func(path)
    elif path.is_dir():
        data = {}
        for cur_path, dirnames, filenames in os.walk(path):
            if '.ipynb_checkpoints' not in cur_path:
                for filename in filenames:
                    filename = pathlib.Path(filename)
                    if filename.suffix == f'.{extension}':
                        cur_path_rel = pathlib.Path(cur_path).relative_to(path)
                        itertools.assign_dict(data, [*cur_path_rel.parts,filename.stem], load_func(os.path.join(cur_path,filename)))
    else:
        raise IOError(f"Path {path} is neither file nor directory")
        
    return data
        
def load_data(path):
    return load(path, 'pkl')

def load_config(path):
    return load(path, 'json')
