import pyarrow.compute as pc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def _is_numeric(dtype):
    rtn = 'int' in dtype
    rtn = rtn or 'float' in dtype
    rtn = rtn or 'double' in dtype
    return rtn

def _bin(array, binning, n_bins):
    '''
    Converts a floating-point pyarrow array into categorical bins based on 
    quantiles. 
    '''
    if binning == 'linear':
        bins = np.arange(n_bins)
    elif binning == 'quantile':
        bins = np.quantile(array, q=[i*1/n_bins for i in range(n_bins+1)])
    return bins

class SupervisedDataset(Dataset):
    def __init__(self, 
        table, 
        metadata, 
        target_column, 
        embed_d_jk, # -1 if no embed => default to 1-hot
        embed_d_cats, # -1 if no embed => default to 1-hot
        inference=False,
        binning=None,
        n_bins=4
        ):

        self.table = table
        self.metadata = metadata
        self.m_fields = metadata['fields']
        self.dense_cols = []
        self.offset_cols = []
        self.target_column = target_column
        self.fields_dense = []
        self.fields_offs = []
        self.field2dim = dict()
        self.field2val2idx = dict()
        self.num_embedding_rows = 0
        self.target_dim = None
        self.dense_len = 0
        self.inference = inference

        for i,field in enumerate(self.m_fields):
            #if field == self.target_column:
            #    continue
                
            if _is_numeric(self.m_fields[field]['dtype']):
                array = np.nan_to_num(self.table[field].to_numpy(),copy=False)
                if binning == None:
                    self.dense_cols.append(
                        torch.tensor(array).float().unsqueeze(1))
                    self.field2dim[field] = 1
                    self.fields_dense.append(field)
                else:
                    bins = _bin(array, binning, n_bins)
                    binned_array = np.digitize(array, bins).astype(str)
                    distincts = np.unique(binned_array)
                    val2idx = {str(val):idx for idx,val in enumerate(distincts)}
                    N = len(val2idx) + 1 # Add one for OOV
                    self.field2val2idx[field] = val2idx
                    ten = torch.tensor(
                        [val2idx.get(val, N-1) for val in binned_array])
                    if embed_d_cats > 0: # Now we treat this as cat
                        self._process_embs(field, ten, N, embed_d_cats)
                        dim = embed_d_cats
                    else:
                        self._process_one_hots(field, ten, N, distincts)
                        dim = N
                    self.field2dim[field] = dim
                        
            else: # Should only be strings but need casting from pyarrow to str
                distincts = self.m_fields[field]['distincts']
                val2idx = {str(val):idx for idx,val in enumerate(distincts)}
                N = len(val2idx) + 1 # Add one for OOV
                self.field2val2idx[field] = val2idx
                if field == metadata['join_field']:
                    embed_d = embed_d_jk 
                else:
                    embed_d = embed_d_cats
                ten = torch.tensor(
                        [val2idx.get(
                            str(val), N-1) for val in self.table[field]])
                if embed_d > 0:
                    self._process_embs(field, ten, N, embed_d)
                    dim = embed_d
                else:
                    self._process_one_hots(field, ten, N, distincts)
                    dim = N
                self.field2dim[field] = dim

        if not inference:
            if _is_numeric(str(self.table.schema[self.table.column_names.index(
                    target_column)].type)):  
                self.target_col = torch.tensor(
                    self.table[target_column].to_numpy()).float().unsqueeze(1)
                self.target_dim = 1
            else:
                distincts = self.m_fields[field]['distincts']
                val2idx = {str(val):idx for idx,val in enumerate(distincts)}
                ten = torch.tensor(
                        [val2idx[str(val)] for val in self.table[field]])
                self.target_col = ten
                self.target_dim = len(val2idx)

        self.dense_cols = torch.concat(self.dense_cols, dim=1)
        self.offset_cols = torch.concat(self.offset_cols, dim=1)
        self.dense_len = self.dense_cols.shape[1]
        self.offset_len = self.offset_cols.shape[1]
        self.fields = self.fields_dense + self.fields_offs 

    def _process_embs(self, field, ten, N, embed_d):
        ten += self.num_embedding_rows
        ten = ten.unsqueeze(1)
        self.num_embedding_rows += N
        self.offset_cols.append(ten)
        self.fields_offs += [f"{field}_emb_{i}" for i in range(embed_d)]

    def _process_numerics(self, col, is_target=False):
        pass

    def _process_one_hots(self, field, ten, N, distincts, is_target=False):
        ten = F.one_hot(ten, num_classes=N)
        if not is_target:
            self.dense_cols.append(ten)
            self.fields_dense += [f"{field}_1hot_{val}" for val in distincts]
            self.fields_dense += [f"{field}_1hot__OOV_"]
        else:
            self.target_col = ten

    def get_dims(self):
        return sum([dim for dim in self.field2dim.values()]), self.target_dim

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        x = (self.dense_cols[idx], self.offset_cols[idx])        
        rtn = x if self.inference else (x, self.target_col[idx])
        return rtn

