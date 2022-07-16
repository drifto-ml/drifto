from datetime import datetime
import json
import duckdb
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Local imports
from .ml.data import SupervisedDataset, _is_numeric
from .ml.models import *

def train(
    feature_table,
    metadata,
    model='linear',
    target='label',
    embed_join_field=True,
    embed_categoricals=False,
    embed_dim=32,
    batch_size=128,
    lr=1e-3,
    precision=32,
    max_epochs=100,
    num_loading_workers=0,
    val_mode=None,
    val_split=None,
    model_export_path=None,
    model_export_format='torch'):
    """
    Run an training job on a given feature_table. Accepts metadata as returned
    by train.

    Parameters
    __________
    feature_table : pl.LightningModule or str
        A table containing samples to train on (as returned by 
        featurize) or path to such a table. 

    metadata : dict
        A dictionary containing metadata as returned by train.

    model : str
        The model type to use

    target : label
        The name of the target column

    embed_join_field : bool
        Turn this flag on to embed the join field into embedding vectors that 
        are then passed into the model as input. 

    embed_categoricals : bool
        Turn this on to embed categorical columns into embedding vectors.

    embed_dim: int
        Embedding dimension 

    batch_size : int
        Batch size to use for inference job.

    lr : float
        Learning rate to use for gradient-based solvers.

    precision : int
        The floating-pint precision to use for inference job.

    num_loeading_workers : int
        The number of loading workers to use for job.

    Returns
    _______
    Predicted values (PyTorch.tensor). Returns values used to predict 
    """


    if type(feature_table) is str: # Assume this is a path
        feature_table = pq.read_table(feature_table)

    if type(metadata) is str: # Assume this is a path
        with open(metadata,'r') as M:
            metadata = json.load(M)

    # Set up embed_d variables to pass into Dataset
    embed_d_jk = embed_dim if embed_join_field else -1
    embed_d_cats = embed_dim if embed_categoricals else -1

    # Process validation split
    if val_mode == None:
        D = SupervisedDataset(feature_table, metadata, target, 
            embed_d_jk, embed_d_cats)
        D_train = D
        D_val = []
    elif val_mode == "random":
        assert type(val_split) == float
        D = SupervisedDataset(feature_table, metadata, target, 
            embed_d_jk, embed_d_cats)
        l = int(len(D)*val_split) 
        D_train, D_val = random_split(D, [len(D)-l, l])
    elif val_mode == "time":
        if type(val_split) == float:
            # Linear interpolate from first to last time period
            max_t = pc.max(feature_table[metadata['time_field']]).as_py()
            min_t = pc.min(feature_table[metadata['time_field']]).as_py()
            time_split_sql = str((1-val_split)*(max_t - min_t) + min_t)
        elif type(val_split) == datetime:
            time_split_sql = str(time_split_sql)
        else:
            raise ValueError
        
        # Time split is easier in SQL at table level
        con = duckdb.connect(database=':memory:')
        train_table = con.execute(
            f"""
                SELECT * FROM feature_table 
                WHERE {metadata['time_field']} <= '{time_split_sql}'
            """).arrow()
        val_table = con.execute(
            f"""
                SELECT * FROM feature_table 
                WHERE {metadata['time_field']} > '{time_split_sql}'
            """).arrow()

        D_train = SupervisedDataset(train_table, metadata, target, 
            embed_d_jk, embed_d_cats)
        D = D_train # Still want this var for consistency with rest of code

        D_val = SupervisedDataset(val_table, metadata, target,
            embed_d_jk, embed_d_cats)
    elif val_mode == 'join':
        raise NotImplementedError
    
    train_loader = DataLoader(D_train, batch_size=batch_size,
            num_workers=num_loading_workers)
    if len(D_val) > 0:
        val_loader = DataLoader(D_val, batch_size=batch_size,
            num_workers=num_loading_workers)
    else:
        val_loader = []

    # Instantiate model
    if model == 'linear':
        model = LinearRegression(*D.get_dims(),
            D.num_embedding_rows, embed_dim, lr=lr)
    elif model == 'logistic':
        model = LogisticRegression(*D.get_dims(),
            D.num_embedding_rows, embed_dim, lr=lr)
    elif model == 'naive_bayes':
        raise NotImplementedError
    else:
        raise ValueError(f'Model type {model} not supported.')

    # Train it, baby!
    trainer = pl.Trainer(precision=precision, max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)

    # Update metadata post-training
    new_metadata = dict()
    new_metadata['featurize'] = metadata
    new_metadata['train'] = {'embed_dim_join_field' : embed_d_jk,
        'embed_dim_categoricals' : embed_d_cats, 'target_column' : target,
        'fields' : D_train.fields}
    metadata = new_metadata

    # Export model if requested
    if model_export_path != None:
        if model_export_format == 'torch':
            torch.save(model, model_export_path)
        elif model_export_format == 'onnx':
            bsize = 1
            x_dense = torch.randn(bsize, D.dense_len, requires_grad=True)
            x_offs = torch.randint(1,(bsize, D.offset_len))
            torch.onnx.export(
                model,
                ((x_dense, x_offs),),
                model_export_path,
                input_names = ['input'],
                output_names = ['output'],
                dynamic_axes={'input' : {0 : 'bsize'},
                            'output' : {0 : 'bsize'}})
    
    return model, metadata
