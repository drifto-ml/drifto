import drifto as dr
import datetime
from pyarrow import parquet as pq
    
if __name__ == '__main__':
    fields = ('user_id', 'timestamp',)
    target_value = 'purchase'
    T = dr.wrangle(*fields, 
        primary_table_path='events.parquet',
        cols=["action", "order_total","attributes->'$.page'"],
        table_paths=[('purchase', 'transactions.parquet')])

    feature_table, inference_table, metadata = dr.featurize('action', 
        *fields, T, 'week', 'action', target_value=target_value,
        histogram_cols=["attributes->'$.page'"],
        binarize_target=True,
        binarize_threshold=0.99,
        filter_inactive=True)

    pq.write_table(feature_table, "features.parquet")

    model, metadata = dr.train(feature_table, metadata, max_epochs=80,
        model='logistic', model_export_path='test.onnx', lr=8e-3,
        val_mode='time',
        val_split=0.8, 
        batch_size=512)

    predicts = dr.inference(model, inference_table, metadata)
