import collections
import re
import logging
import sys

import torch
import numpy as np
import numpy.ma as ma
import pandas as pd

from .. import itertools, timeit
from .. import np as unp
from . import array as ar
from . import parsing

logger = logging.getLogger(__name__)

__all__ = ['NA', 'InternalError', 'DataFrame', 'DataFrameGroupBy', 'concat']

NA = ma.masked

class InternalError(Exception):
    pass
    
def _is_0d_col_idx(obj):
    return isinstance(obj, collections.abc.Hashable) or (isinstance(obj, np.ndarray) and obj.ndim == 0) or (isinstance(obj, torch.Tensor) and obj.ndim == 0)

def _is_1d_col_idx(obj):
    return isinstance(obj, list) or isinstance(obj, slice) or (isinstance(obj, np.ndarray) and obj.ndim == 1) or (isinstance(obj, torch.Tensor) and obj.ndim == 1)

def _is_0d_row_idx(obj):
    return isinstance(obj, int) or (isinstance(obj, np.ndarray) and obj.ndim == 0) or (isinstance(obj, torch.Tensor) and obj.ndim == 0)

def _is_1d_row_idx(obj):
    return isinstance(obj, list) or isinstance(obj, slice) or (isinstance(obj, np.ndarray) and obj.ndim == 1) or (isinstance(obj, torch.Tensor) and obj.ndim == 1)
    
class DataFrame:
    def __init__(self, data, columns=None, copy=True):
        """
        Initialize a DataFrame with the given data and columns.
        Data is stored in the .data attribute and is a dictionary of ar.Array representing the columns.
        If copy is True, constructs a copy of data (arrays and tensor are copied, but for objects, only references are copied).
        Otherwise, only dictionary references (if data is a dictionary) are copied.
        
        When indexing:
            - an element is returned as a 0d torch.Tensor, 0d np.ndarray, or object
            - a column is returned as a 1d torch.Tensor or np.ndarray
            - a row is returned as a dictionary of 0d torch.Tensor, 0d np.ndarray, or object
            - a sub-frame is returned as a DataFrame

        :param data: a 2D np.ndarray/torch.Tensor, a dictionary of (0/1D torch.Tensor/np.ndarray or any other object), or a DataFrame
        :param columns: a list of column names (optional)
        """
        if columns is not None:
            if not (isinstance(columns, collections.abc.Iterable) and isinstance(columns, collections.abc.Sized)) or isinstance(columns, np.ndarray) or isinstance(columns, torch.Tensor):
                raise TypeError(f"columns must be None or be a sized iterable, but {type(columns)=}.")
                
            if len(set(columns)) != len(columns):
                raise ValueError(f"Columns must be unique, but detected duplicates in {columns}.")
        
        if isinstance(data, dict):
            if columns is not None:
                raise ValueError(f"columns must be None when data is a dictionary, but {type(columns)=}.")

            arrs, shapes = [], []
            for v in data.values():
                arr = ar.Array(v)
                if arr.ndim > 1:
                    raise ValueError(f"data must be at most 1D, but {arr.ndim=}.")
                    
                arrs.append(arr)
                shapes.append(arr.shape)
                
            shape = np.broadcast_shapes(*shapes)
            
            if copy:
                data = {k: v.broadcast_to(shape).copy() for k, v in zip(data.keys(), arrs)}
            else:
                data = {k: v.broadcast_to(shape) for k, v in zip(data.keys(), arrs)}
            
        elif isinstance(data, DataFrame):
            if columns is not None:
                raise ValueError(f"column must be None when data is a DataFrame, but {type(columns)=}.")
                
            if copy:
                data = {k: v.copy() for k, v in data._items()}
            else:
                data = data._data.copy() # dictionary shallow copy
            
        elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            if data.ndim != 2:
                raise ValueError(f"data must be 2D if it is a np.ndarray or torch.Tensor, but {data.ndim=}.")
                
            if columns is None:
                columns = range(data.shape[1])

            if len(columns) != data.shape[1]:
                raise ValueError(f"Number of columns must be the second dim of data, but {len(columns)=}, {data.shape[1]=}.")
            
            data = ar.Array(data)
            if copy:
                data = data.copy()
            data = {column: data[:,i] for i, column in enumerate(columns)}
        
        else:
            raise TypeError(f"data must be a 2D list/np.ndarray/torch.Tensor, a dictionary of (0/1D torch.Tensor/np.ndarray or any other object), or a DataFrame, but got {type(data)}.")
                
        self._data = data
        
    def _validate(self):
        lengths = {}
        for column, data in self._items():
            if not isinstance(ar.Array):
                raise InternalError(f"column data must be ar.Array, but {type(arr)=}.")
            if not data.ndim == 1:
                raise InternalError(f"column data must be 1D, but {data.ndim=}.")
            lengths[column] = len(series)

        if not unp.isconst(lengths.values()):
            raise InternalError(f"each column must have the same length, but {lengths=}.")
        
    @property
    def columns(self):
        return list(self._data.keys())
    
    def __len__(self):
        if len(self._data) == 0:
            return 0
        return len(next(iter(self._data.values())))
    
    @property
    def shape(self):
        return (len(self), len(self._data))
    
    def keys(self):
        yield from self._data.keys()
        
    def _values(self):
        yield from self._data.values()
    
    def values(self):
        for v in self._data.values():
            yield v.data
            
    def _items(self):
        yield from self._data.items()
    
    def items(self):
        for k, v in self._data.items():
            yield k, v.data
        
    def iterrows(self):
        for i in range(len(self)):
            yield i, self[:,i]
     
    @property
    def dtype(self):
        return {k: v.dtype for k, v in self._items()}
    
    @property
    def device(self):
        return {k: v.device if v.is_tensor else None for k, v in self._items()}
    
    @property
    def is_tensor(self):
        return {k: v.is_tensor for k, v in self._items()}
    
    @property
    def nbytes(self):
        return {k: v.nbytes for k, v in self._items()}

    def __getitem__(self, key):
        """
        Index the DataFrame as if indexing a 2D array with arr[:, col_idx][row_idx], where key = (col_idx, row_idx).
        Returns a 0d array if both col_idx and row_idx are 0d, a list of 0d elements if row_idx is 0d,
        a 1d np.ndarray/torch.Tensor if col_idx is 0d, and a dataframe if neither is 0d.
        Note that it is different from indexing a 2D array with arr[col_idx, row_idx] when both col_idx and row_idx are array-like.
        col_idx can only be a slice if it is slice(None) or slice(None, None, step), since it would introduce 
        ambiguity as to whether to interpret slice.start and slice.end as column indices or as column names.
        
        There is no need to worry about making row_idx the same object type and device as column (this would be impossible if
        indexing multiple columns with different object types and devices) - Array.__getitem__ takes care of this issue.
        """
        if isinstance(key, tuple):
            col_idx, row_idx = key
        else:
            col_idx, row_idx = key, slice(None)
            
        if col_idx is Ellipsis:
            col_idx = slice(None)
            
        if row_idx is Ellipsis:
            row_idx = slice(None)
        
        if _is_0d_col_idx(col_idx):
            return self._data[col_idx][row_idx].data
        
        if _is_1d_col_idx(col_idx):
            if isinstance(col_idx, slice):
                if col_idx.start is not None or col_idx.stop is not None:
                    raise ValueError(f"col_idx can only be a slice if both start and end are None, but {col_idx=}")
                    
                col_idx = self.columns[col_idx]

            if _is_0d_row_idx(row_idx):
                return [self._data[k][row_idx].data for k in col_idx]
            
            if _is_1d_row_idx(row_idx):
                return DataFrame({k: self._data[k][row_idx].data for k in col_idx})
            
            raise TypeError(f"Invalid row index {type(row_idx)=}.")

        raise TypeError(f"Invalid column index {type(col_idx)=}.")
        
    def _set_column(self, column, row_idx, arr):
        # NOTE: row_idx = None produces different behavior than row_idx = slice(None)
        # even though both sets all elements of a column
        if row_idx is None:
            self._data[column] = arr.broadcast_to(len(self))
        else:
            if not column in self.columns:
                self._data[column] = ar.Array.empty(
                    (len(self),),
                    is_tensor=arr.is_tensor,
                    dtype=arr.dtype,
                    device=arr.device,
                )
            if not self._data[column].writeable:
                self._data[column] = self._data[column].copy()
            if self._data[column].is_tensor:
                arr = arr.astensor() # allow setting tensor elements with np array
            self._data[column][row_idx] = arr
        
    def __setitem__(self, key, value):
        """
        Set elements of the DataFrame. One can understand it abstractly as doing
        arr.T[col_idx, :][row_idx] = value, where key = (col_idx, row_idx), except
        for two major differences:
            1) In numpy this assignment would not modify the array, but here it would.
            2) In numpy the value is broadcasted to the correct shape by adding leading batch dimensions,
               but here the value is broadcasted to the correct shape by adding trailing batch dimensions.
               
        There is no need to worry about making row_idx the same object type and device as column (this would be impossible if
        indexing multiple columns with different object types and devices) - Array.__setitem__ takes care of this issue.
        """
        if isinstance(key, tuple):
            col_idx, row_idx = key
        else:
            col_idx, row_idx = key, None
            
        if col_idx is Ellipsis:
            col_idx = slice(None)
            
        if row_idx is Ellipsis:
            row_idx = slice(None)
            
        value = ar.Array(value)
        
        if _is_0d_col_idx(col_idx):
            self._set_column(col_idx, row_idx, value)
            
        elif _is_1d_col_idx(col_idx):
            if isinstance(col_idx, slice):
                if col_idx.start is not None or col_idx.stop is not None:
                    raise ValueError(f"col_idx can only be a slice if both start and end are None, but {col_idx=}")
                    
                col_idx = self.columns[col_idx]

            # broadcast value to (len(col_idx), ...) so that we can iterate over columns.
            # this broadcasting different form the usual numpy broadcasting rule in that 
            # batch dimensions here are TRAILING not leading.
            if value.ndim <= 1:
                value = value.broadcast_to(len(col_idx))
            elif value.ndim == 2:
                value = value.broadcast_to((len(col_idx), -1))
            else:
                raise ValueError(f"value must be at most 2D, but {value.ndim=}")
                
            for k, v in zip(col_idx, value):
                self._set_column(k, row_idx, ar.Array(v))
    
    def __repr__(self):
        return repr(self.to_pandas())
        
    def copy(self):
        # Should work similarly to pd.DataFrame.copy(deep=True)
        return DataFrame(self._data, copy=True)
        
    def to(self, device):
        df = self.copy()
        for k, v in df._items():
            df[k] = v.to(device)
        return df
    
    def to_list(self):
        """
        Returns data as a list of columns. Data is copied (but non-array objects are not copied).
        """
        return list(self.copy().values())
    
    def to_dict(self):
        """
        Returns data as a dict of columns. Data is copied (but non-array objects are not copied).
        """
        return self.copy()._data
    
    def to_numpy(self):
        """
        Returns data as a 2D np.ndarray. Data is copied (but non-array objects are not copied).
        """
        return np.stack([v.detach().cpu().numpy() for v in self._values()], axis=1) # np.stack returns copy
    
    def to_torch(self, device=None):
        """
        Returns data as a 2D torch.Tensor, but may raise error if there is a column with a datatype
        that is not supported by pytorch. Data is copied (but non-array objects are not copied).
        """
        return torch.stack([v.astensor().data.to(device) for v in self._values()], dim=1)
    
    def to_pandas(self, copy=True):
        """
        Returns data as a pd.DataFrame.
        """
        return pd.DataFrame({k: v.detach().cpu().numpy() for k, v in self._items()}, copy=copy)

    # def to_html(self, *args, **kwargs):
    #     return self.to_pandas(copy=False).to_html(*args, **kwargs)
    
    def to_html(self, max_rows=None, show_dimensions=False, formatters=None, show_column_info=True):
        """
        chatGPT-assisted code. Slightly slower than using the pd.DataFrame.to_html,
        but allows for more control over display detail, since we don't need to convert
        torch.Tensor to np.ndarray.
        """
        if max_rows is None:
            max_rows = len(self)
        else:
            max_rows = min(max_rows, len(self))
        
        if len(self) <= max_rows:
            row_indices = range(len(self))
        else:
            n_top = max_rows // 2
            n_bottom = max_rows - n_top
            row_indices = list(range(n_top)) + [-1] + list(range(len(self) - n_bottom, len(self)))
        
        html = "<table><thead><tr>"
        html += "<th></th>"
        
        for column in self.columns:
            if show_column_info:
                is_tensor, dtype, device = self._data[column].is_tensor, str(self._data[column].dtype), self._data[column].device
                otype = 'torch' if is_tensor else 'numpy'
                search = re.search('.*\.(.*)', dtype)
                dtype = dtype if search is None else search.group(1)
                device = 'cpu' if device is None else str(device)
                header = [column, otype, dtype, device]
            else:
                header = [column]
            html += f"<th>{'<br/>'.join(header)}</th>"
        html += "</tr></thead><tbody>"
        
        for i in row_indices:
            if i == -1:
                html += "<tr>"
                html += "<td>...</td>" * (len(self.columns) + 1)
                html += "</tr>"
            else:
                html += "<tr>"
                html += f"<td><b>{i}</b></td>"
                for column in self.columns:
                    cell = self[column][i]
                    if formatters is not None and column in formatters:
                        cell = formatters[column](cell)
                    
                    html += f"<td>{cell}</td>"
                html += "</tr>"
        html += "</tbody></table>"
        
        if show_dimensions:
            html += f"<p>{len(self)} rows × {len(self.columns)} columns</p>"
        
        return html
    
    def _to_numpy_numeric(self):
        arrs = []
        for arr in self._values():
            arr = arr.detach().cpu().numpy() # do everything in numpy due to various torch.unique bugs
            if arr.dtype.kind not in 'biufc': # non-numeric
                _, arr = np.unique(arr, return_inverse=True)
            if isinstance(arr, ma.MaskedArray):
                arr = arr.astype(np.result_type(arr.dtype, float)) # promote to at least float so we can turn masked elements to nan
                arr[arr.mask] = np.nan
                arr = arr.data # drop mask, turns ma.MaskedArray into np.ndarray
            arrs.append(arr)
        arr = np.stack(arrs, axis=-1)
        return arr
    
    def info(self):
        return pd.DataFrame([{
            'column': k,
            'n_rows': len(v), # should be same for every column
            'is_tensor': v.is_tensor,
            'dtype': v.dtype,
            'device': v.device,
            'nbytes': v.nbytes,
        } for k, v in self._items()])
    
    def __delitem__(self, columns):
        columns = np.atleast_1d(columns)
        for column in columns:
            del self._data[column]
    
    def drop(self, *columns, copy=False):
        df = DataFrame(self, copy=copy)
        for column in columns:
            del df[column]
        return df
    
    def rename(self, kwargs, copy=False):
        return DataFrame({kwargs.get(k, k): self._data[k] for k in self._data}, copy=copy) # preserves order
    
    def groupby(self, *args, **kwargs):
        # Should works similarly to pd.DataFrame.groupby(by, as_index=False)
        return DataFrameGroupBy(self, *args, **kwargs)
    
    def where(self, condition, level=0):
        var_names = parsing.parse_var_names(condition)
        condition = parsing.modify_expr(condition, var_names)
        frame = sys._getframe(level+1) # see https://github.com/pandas-dev/pandas/blob/main/pandas/core/computation/scope.py
        # add @ variables
        var_dict = {f'__eval_local_{var_name}': frame.f_locals[var_name] for var_name in var_names}
        # add dataframe columns
        var_dict = var_dict | {k: v.data for k, v in self._data.items()}
        # add ability to recognize backtick quoted columns
        var_dict = var_dict | {f"BACKTICK_QUOTED_STRING_{k.replace(' ', '_')}": v.data for k, v in self._data.items()}
        result = eval(condition, var_dict)
        return result
    
    def query(self, condition, level=0):
        return self[:,self.where(condition, level=level+1)]
    
    def merge(self, df, how='inner', on=None, left_on=None, right_on=None, suffixes=('_x', '_y')):
        # if not isinstance(df, DataFrame):
        #     raise TypeError(f"df must be a DataFrame, but {type(df)=}.")
        
        if not how == 'inner':
            raise NotImplementedError()
        
        if on is not None:
            if not (left_on is None and right_on is None):
                raise ValueError(f"left_on and right_on must be None when on is not None, but {left_on=}, {right_on=}.")
            left_on, right_on = on, on
            
        if (left_on is None and right_on is not None) or (left_on is not None and right_on is None):
            raise ValueError(f"left_on and right_on must either be both None or not None, but {left_on=}, {right_on=}.")
        
        if left_on is None: # right_on must also be None
            on = [column for column in self.columns if column in df.columns] # same order as left df
            left_on, right_on = on, on
        else:
            left_on, right_on = np.atleast_1d(left_on), np.atleast_1d(right_on)
            if len(left_on) != len(right_on):
                raise ValueError(f"Length of left_on must be same as length of right_on, but {len(left_on)=}, {len(right_on)=}.")
          
        combined = concat([self[left_on], df[right_on].rename({r: l for l, r in zip(left_on, right_on)})])._to_numpy_numeric()
        larr, rarr = combined[:len(self)], combined[len(self):]
        
        larr, linv, lcount = unp.unique_rows(larr, sorted=False, return_inverse=True, return_counts=True)
        rarr, rinv, rcount = unp.unique_rows(rarr, sorted=False, return_inverse=True, return_counts=True)
        # logger.debug(f'{larr=}')
        # logger.debug(f'{linv=}')
        # logger.debug(f'{lcount=}')
        # logger.debug(f'{rarr=}')
        # logger.debug(f'{rinv=}')
        # logger.debug(f'{rcount=}')
        _, larr_idx, rarr_idx = unp.intersect_rows(larr, rarr, assume_unique=True, return_indices=True)
        # logger.debug(f'{larr_idx=}')
        # logger.debug(f'{rarr_idx=}')
        lmask, rmask = np.isin(linv, larr_idx), np.isin(rinv, rarr_idx)
        # logger.debug(f'{lmask=}')
        # logger.debug(f'{rmask=}')
        lindices, rindices = lmask.nonzero()[0], rmask.nonzero()[0]
        # logger.debug(f'{lindices=}')
        # logger.debug(f'{rindices=}')
        inv_larr_idx, inv_rarr_idx = np.empty(len(larr), dtype=int), np.empty(len(rarr), dtype=int)
        inv_larr_idx[larr_idx] = np.arange(len(larr_idx))
        inv_rarr_idx[rarr_idx] = np.arange(len(rarr_idx))
        # logger.debug(f'{inv_larr_idx=}')
        # logger.debug(f'{inv_rarr_idx=}')
        lindices, rindices = lindices[np.argsort(inv_larr_idx[linv[lmask]])], rindices[np.argsort(inv_rarr_idx[rinv[rmask]])]
        # logger.debug(f'{lindices=}')
        # logger.debug(f'{rindices=}')
        lcount, rcount = lcount[larr_idx], rcount[rarr_idx]
        # logger.debug(f'{lcount=}')
        # logger.debug(f'{rcount=}')
        lindices = np.repeat(lindices, np.repeat(rcount, lcount))
        rindices = unp.repeat(rindices, lcount, chunks=rcount)
        # logger.debug(f'{lindices=}')
        # logger.debug(f'{rindices=}')
        
        left, right = self[:,lindices], df[:,rindices]
        del right[[column for column in left_on if column in right_on]]
        left_rename = {column: f'{column}{suffixes[0]}' for column in left.columns if column in right.columns}
        right_rename = {column: f'{column}{suffixes[1]}' for column in right.columns if column in left.columns}
        
        return concat([left.rename(left_rename), right.rename(right_rename)], axis=1)
    
class DataFrameGroupBy:
    def __init__(self, df, by, sort=True, dropna=True):
        assert isinstance(df, DataFrame)
        self.df = df
        self.by = np.atleast_1d(by)
        self.dropna = dropna
        data = self.df[self.by]
        
        if len(data.columns) == 0:
            raise ValueError("No selected columns.")
            
        arr = self.df[self.by]._to_numpy_numeric() # numeric numpy array representation of selected columns
        _, idx, inv_idx = unp.unique_rows(arr, sorted=sort, return_index=True, return_inverse=True) # faster than np.unique
        # _, idx, inv_idx = np.unique(arr, return_index=True, return_inverse=True, axis=0)
        self._groups = DataFrame({column: self.df[column][idx] for column in self.by})
        self._na = np.stack([arr.isna().detach().cpu().numpy() for arr in self._groups._values()], axis=1).any(axis=1)
        self._row_to_group_idx = ar.Array(inv_idx)
        
    @property
    def groups(self):
        if self.dropna:
            return self._groups[:,~self._na]
        return self._groups
    
    def items(self):
        columns = [column for column in self.df.columns if column not in self.by]
        for idx, group in self._groups.iterrows():
            if self.dropna and self._na[idx]:
                continue
                
            yield tuple(group), self.df[columns, self._row_to_group_idx.data == idx]
        
    def _agg(self, column, func, results):
        if (column, func) in results:
            return results[(column, func)]
        
        _row_to_group_idx = self._row_to_group_idx.like(self.df[column], dtype=False)

        # See scipy.stats.binned_statistic_dd for reference
        if func == 'count':
            result = _row_to_group_idx.bincount()
        
        elif func == 'sum':
            result = _row_to_group_idx.bincount(weights=self.df[column])
        
        elif func == 'mean':
            result = self._agg(column, 'sum', results) / self._agg(column, 'count', results)
            
        elif func == 'var':
            count, mean = self._agg(column, 'count', results), self._agg(column, 'mean', results)
            result = (_row_to_group_idx.bincount(
                weights=(self.df[column] - mean[_row_to_group_idx].data)**2
            ))/(count-1)
            
        elif func == 'std':
            var = self._agg(column, 'var', results)
            result = var**0.5
        
        elif func == 'sem':
            count, std = self._agg(column, 'count', results), self._agg(column, 'std', results)
            result = std / count**0.5
        
        elif func == 'min':
            v = ar.Array(self.df[column])
            i = v.argsort(descending=True)
            result = ar.Array.empty((len(self._groups),), masked=False, is_tensor=v.is_tensor, dtype=v.dtype, device=v.device)
            
            # In pytorch calling __setitem__ with an index tensor that has duplicate elements is undefined and non-deterministic by                 # default. See the documentation on index_put_. In my own experiments, setting use_deterministic_algorithms resolves this               # issue and makes pytorch behave the same way as numpy, but I don't think this behavior is gauranteed.
            if set_enabled := v.is_tensor and not torch.are_deterministic_algorithms_enabled():
                torch.use_deterministic_algorithms(True)
            result[_row_to_group_idx[i]] = v[i] 
            if set_enabled:
                torch.use_deterministic_algorithms(False)
        
        elif func == 'max':
            v = ar.Array(self.df[column])
            i = v.argsort()
            result = ar.Array.empty((len(self._groups),), masked=False, is_tensor=v.is_tensor, dtype=v.dtype, device=v.device)
            
            # In pytorch calling __setitem__ with an index tensor that has duplicate elements is undefined and non-deterministic by                 # default. See the documentation on index_put_. In my own experiments, setting use_deterministic_algorithms resolves this               # issue and makes pytorch behave the same way as numpy, but I don't think this behavior is gauranteed.
            if set_enabled := v.is_tensor and not torch.are_deterministic_algorithms_enabled():
                torch.use_deterministic_algorithms(True)
            result[_row_to_group_idx[i]] = v[i] 
            if set_enabled:
                torch.use_deterministic_algorithms(False)
        
        elif isinstance(func, callable):
            v = ar.Array(self.df[column])
            result = ar.Array.empty((len(self._groups),), masked=False, is_tensor=v.is_tensor, dtype=v.dtype, device=v.device)
            for idx, group in enumerate(self._groups):
                result[idx] = func(v[_row_to_group_idx == idx].data)
        
        else:
            raise ValueError(f"func must be a callable or a string in ['count', 'sum', 'mean', 'var', 'std', 'sem', 'min', 'max'], but got {func}")
        return result
        
    def agg(self, **kwargs):
        data = {k: v for k, v in self.groups.items()}
        results = {}
        for _, v in kwargs.items():
            results[tuple(v)] = self._agg(*v, results)
            
        results = {k: v[~self._na].data if self.dropna else v.data for k, v in results.items()}
        data = data | {k: results[v] for k, v in kwargs.items()}
        
        return DataFrame(data)
        
def concat(dfs, axis=0):
    assert len(dfs) > 0
    all_columns = [df.columns for df in dfs]
    
    if axis == 0:
        if not unp.isconst(all_columns, axis=0).all():
            raise ValueError(f"dfs must have the same columns when axis=0, but {all_columns=}.")
            
        columns = all_columns[0]
        return DataFrame({column: ar.concat([df[column] for df in dfs]) for column in columns})
    
    elif axis == 1:
        columns = itertools.flatten_seq(all_columns)
        if len(set(columns)) != len(columns):
            raise ValueError(f"dfs must have non-overlapping columns when axis=1, but {all_columns=}.")
            
        return DataFrame({k: v for df in dfs for k, v in df.items()})
    
    raise ValueError("axis must be 0 (concatenate along rows) or 1 (concatenate along columns)")