import json
import pyarrow.parquet as pq
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Local imports
from .ml.dataloaders import SupervisedDataset

def inference(
    model,
    inference_table,
    metadata,
    batch_size=128,
    precision=32,
    num_loading_workers=0):
    """
    Run an inference job on a given inference_table. Accepts metadata as returned by train.

    Parameters
    __________
    model : pl.LightningModule or str
        A trained model (as returned by train) or a path to an exported model. 

    inference_table : pyarrow.Table or str
        A table containing samples to run inference on (as returned by 
        featurize) or path to such a table.

    metadata : dict
        A dictionary containing metadata as returned by train.

    batch_size : int
        Batch size to use for inference job.

    precision : int
        The floating-pint precision to use for inference job.

    num_loeading_workers : int
        The number of loading workers to use for job.

    Returns
    _______
    Predicted values (PyTorch.tensor). Returns values used to predict 
    """

    if type(inference_table) is str: # Assume this is a path
        inference_table = pq.read_table(inference_table)

    if type(metadata) is str: # Assume this is a path
        with open(metadata,'r') as M:
            metadata = json.load(M)

    embed_d_jk = metadata['train']['embed_dim_join_field']
    embed_d_cats = metadata['train']['embed_dim_categoricals']
    target = metadata['train']['target_column']
    D = SupervisedDataset(inference_table, metadata['featurize'], 
        target, embed_d_jk, embed_d_cats, inference=True)

    inference_loader = DataLoader(D, batch_size=batch_size,
            num_workers=num_loading_workers)

    trainer = pl.Trainer(precision=precision, max_epochs=1)
    predicts = trainer.predict(model, inference_loader)
    return predicts
