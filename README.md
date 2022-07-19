![Drifto](https://uploads-ssl.webflow.com/62620d17539e43f8929e8001/626342c33db7194f75cde160_Drifto-p-500.png)

# [Drifto](https://www.driftoml.com): Automatic Featurization :robot: for User Event Data :busts_in_silhouette: 

User event data (clickstream, transactions, product interactions, etc.) is one of the highest volume and veracity data sources collected by organizations, but it is still notoriously hard to featurize event streams and generate data-driven insights or actionable models.

Drifto is an automated feature engineering and machine learning tool. Drifto automatically generates a large number of user-centric *autofeatures* over a specified time period. Drifto offers a nearly fully-automated *point-and-shoot* experience: just point Drifto towards your raw event tables! Drifto also provides a suite of machine learning models that automatically interoperate with your generated feature tables.

Drifto is built on [DuckDB](https://duckdb.org/) and [Apache Arrow](https://arrow.apache.org/docs/index.html), and therefore is scalable to large datasets. Contact us at `founders@driftoml.com` if you are interested in scaling Drifto up to the petabyte scale with a fully-managed cloud deployment. 

### Drifto Can Automatically :zap: : 
- Join, merge, and wrangle disparate user event tables across all user touch points
- Generate dozens, hundreds, or even thousands of ML-ready *autofeatures* 
- Train models on training features and run inference on production features
- [soon] Schedule and manage your Drifto pipelines to keep tables and models updated
- [soon] Track data lineage all the way from raw data to processed features to trained models.
- [soon] Combine with self-supervised *deep neural autofeatures* that allow for unprecedented levels of user-behavior understanding

### Drifto's Top Workflows :trophy: :
- Customer Value Estimation
- Churn Prediction
- Anomaly Detection
- [soon] Personalization
- [soon] Demand Sensing 

### Quick Start

Install Drifto with `pip install drifto`.

### Example

See the `examples` directory for our primary example.
The sample data has two tables, one with website
clickstream data (`events.parquet`) and one with checkout transactions (`transactions.parquet`).
The example merges the two tables into one master event table with `drifto.wrangle` and then
uses `drifto.featurize` to automatically compute a large number of features for each user for
each week based on different aggregations of the 'action', 'page', and other columns. These features
are used to predict whether a user will stop making purchases in the subsequent week. See
[the docs](docs/doc.md) for a more detailed example walkthrough.

```python
fields = ('user_id', 'timestamp',)
T = drifto.wrangle(*fields, 
    primary_table_path='events.parquet',
    cols=["action", "order_total","attributes->'$.page'"],
    table_paths=[('purchase', 'transactions.parquet')])

feature_table, inference_table, metadata = drifto.featurize('action', 
    *fields, T, 'week', 'action', target_value='purchase',
    histogram_cols=["attributes->'$.page'"],
    filter_inactive=True)

pq.write_table(feature_table, "features.parquet")

model, metadata = drifto.train(feature_table, metadata, max_epochs=80,
    model='logistic', model_export_path='test.onnx', lr=8e-3, 
    batch_size=512)

predicts = drifto.inference(model, inference_table, metadata)
```

`events.parquet` has the following schema:

```
user_id: int64
timestamp: timestamp[ms]
action: string
attributes: string
```

where the `action` column specifies the type of action taken and `attributes` is a
JSON object with action-specific data like the particular page visited for a `page_visited`
action.

`transactions.parquet` has the following schema:

```
user_id: int64
timestamp: timestamp[ms]
order_total: int64
```

The merged event table produced by `wrangle` has the following schema:

```
timestamp: timestamp[ms]
user_id: string
action: string
order_total: int64
attributes_page: string
```

where `order_total` comes from the `transactions` table and `attributes_page` is extracted from the `attributes` JSON in `events.parquet` and is the particular page visited for a `page_visited` action (and `null` for other action types).

Finally, the feature table has the following schema:

```
user_id: string
time_period: timestamp[us]

action_page_visited_count: int64
action_page_visited_count_1: int64
action_email_opened_count: int64
action_email_opened_count_1: int64
...
attributes_page_payments_count: int64
attributes_page_payments_count_1: int64
...
action_count_distinct: int64
action_count_distinct_1: int64
action_mode: string
action_mode_1: string
...
attributes_page_count_distinct: int64
attributes_page_count_distinct_1: int64
...
order_total_sum: double
order_total_sum_1: double
order_total_avg: double
order_total_avg_1: double
order_total_std: double
order_total_std_1: double

label: bool
```

where the first group of features counts the number of occurrences of each action type for each user for
each time period. The features with `_1` affixed to the end are from the previous time period for the
same user. The second set of features counts the number of occurrences of each particular viewed page
for `page_viewed` events. The third and fourth sets of features are the standard ones computed for 
all categorical columns. Finally, the last set of features show the standard ones computed for numerical
columns.
