## Drifto Docs

### Overview

There are only a few core concepts for a Drifto developer to think about. First, we have a `join_field`. For most applications, the `join_field` will be the user ID field (other possibilites include, store ID, item ID, etc.). The `join_field` represents the central concept that Drifto should use to group individual events by.

Second, we have the `time_field`. This is basically exactly what is sounds like -- it is just the timestamp field for the individual events.

Third, we have an `event`. In Drifto, an `event` is any record in any table (of essentially arbitrary schema) that contains the `join_field` and the `time_field`. Any table containing `event` records is an *event table*. These `event_tables` are the main inputs to Drifto's automatic featurization pipeline.

Fourth, we have `cols`. In Drifto, `cols` denote the names of the columns across all of the `event_tables` that Drifto should pay attention to.

Finally, we have the `time_period`. This is time period over which Drifto should run its core feature aggregations. For example, if `time_period` is set to `week`, then Drifto provides a feature row for each distinct value of `join_field` over each `week`.

Once you specify the `join_field`, `time_field`, and some `event_tables` with some `cols`, you can let Drifto's automatic pipeline do the rest! Currently, Drifto's pipeline is comprised of only four API calls: `wrangle`, `featurize`, `train`, and `inference`.

While Drifto is designed to be plug-and-play, you may desire finer-grained control of the pipeline by setting a few flags or tuning a knob or two. If so, read on to the documentation to learn how to do that with Drifto's API calls.

### Detailed Docs

In a Python interpreter shell, if you `import drifto`, you can run `help(drifto.wrangle)`,
`help(drifto.featurize)`, `help(drifto.train)`, and `help(drifto.inference)` to see detailed
docs for each of our four API calls.

### Detailed Example Walkthrough

We provide a sample data generator, `gen_user_events.py`, that generates user interaction
events for a hypothetical website. If the user encounters a buggy page ("/bug"), their
probability of making a purchase goes to zero for the following week.
We aim to recover this fact from the raw event data.

After installing Drifto, `cd` to the `examples` directory. Run `python gen_user_events.py -u 100 -w 100 -a 10`
to generate data for 100 users with 10 events per week for 100 weeks. `events.parquet` stores basic
user actions -- 'page\_viewed', 'email\_opened', and 'search\_performed' -- along with event-specific
metadata, such as the particular page viewed for a 'page\_viewed' event. `transactions.parquet` stores
all user purchases along with the total amount spent on the purchase.

We now walk through the key code in `example.py`.

```python
import drifto as dr

table = dr.wrangle('user_id', 'timestamp',
    primary_table_path='events.parquet',
    cols=["action", "order_total","attributes->'$.page'"],
    table_paths=[('purchase', 'transactions.parquet')])
```

Drifto's `wrangle` call joins together multiple event tables into a single event
table to featurize.
`wrangle` takes the names of the join column and time column (which should be present in all tables), the
name of the primary events table (`events.parquet`), other tables to incorporate into the
merged events table (in this case `transactions.parquet`, whose rows will be merged in with
an event type of 'purchase'), and the columns in the merged table to use to generate features (here `order_total`
from the transactions table and the `page` column from the `attributes` JSON in the primary events table,
specified with a DuckDB JSON selector).

```python
feature_table, inference_table, metadata = dr.featurize('action', 
    'user_id', 'timestamp', table, 'week', 'action',
    target_value='purchase',
    histogram_cols=["attributes->'$.page'"],
    filter_inactive=True)
```

The next call featurizes the merged event table. The same main columns are passed in, as well as the
period to featurize on (`'week'`, which means that there will be a row of features for each week
for each user), the target column (`'action'`), and the value in the target column whose count for
the subsequent period to use as the target (`'purchase'`, meaning that the target is the total count of
purchase events for this user for the subsequent week). The `histogram_cols` are other categorical columns
(the event type column, `'action'`, is treated this way by default) for which the count of each value should be
turned into a feature (categorical columns not specified this way will simply have their mode and distinct count
turned into features). In this case, a JSON selector is used to select the page the user looked at for a `'page_viewed'`
action, so the view count for each distinct page on the website for each user week will be a feature. For numerical
columns, standard features such as the `sum`, `average`, and `variance` are computed.

The final parameter,
`filter_inactive`, indicates that the returned feature and inference tables should be composed only of users that
have had a nonzero target in one of the current period or previous `n_periods_per_row_for_features` periods
(`n_periods_per_row_for_features` determines how many previous periods' features are included in the current row,
and is set to 1 by default). By filtering to active users, the churn prediction problem becomes more interesting, since
we are trying to predict which currently active users will drop off. `featurize` returns a training feature table,
an inference table
comprised of the final week of features for each user, as well as metadata about each column used by
ML model training.

The final two calls in the example train a model on the training data and then make predictions for the target
on the final week of features for each user.
