import duckdb
import pyarrow as pa
import pyarrow.compute as pc

# Local imports
from .ml.data import _is_numeric
from .wrangle import _clean_name

numeric_types = {
    pa.int8(), 
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.float16(),
    pa.float32(),
    pa.float64(),
    pa.bool_()
}

categorical_types = {
    pa.int8(), 
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.string(),
    pa.bool_()
}

def featurize(
    event_col,
    join_field,
    time_field,
    wrangled_table,
    time_period,
    target_column,
    target_value=None,
    events=None,
    histogram_top_k=50,
    histogram_cols=None,
    top_k=10,
    drop_null=True,
    n_time_periods_per_row_for_features=1,
    n_time_periods_per_row_for_target=1,
    filter_inactive=False,
    binarize_target=False,
    binarize_threshold=0):
    """
    Featurize a wrangled event table. Compute features for each entity
    in `join_field` for each `time_period`. For numerical columns, sum,
    average, and variance are computed for the current time period and previous
    `n_time_periods_per_row_for_features` periods. For categorical columns,
    the count of distinct values and the mode value are computed for the
    current time period and previous periods.
    For the categorical columns in `histogram_cols`, features are created for the
    counts of the specified values or of the top `histogram_top_k` most frequent
    values in the columns. Finally, a target is computed for each row by summing the
    value of `target_column` for the next `n_time_periods_per_row_for_target` periods if
    `target_column` is numerical, otherwise by summing the count of the `target_value`
    value within `target_column` for the next periods.

    Parameters
    __________
    event_col : str
        Name of column containing the type of each event.

    join_field : str
        Name of column containing logical entities to group by (e.g. user ID column).

    time_field : str
        Name of column containing event timestamp.

    wrangled_table : pyarrow.Table
        A merged event table returned by drifto.wrangle().

    time_period : str
        The desired time period over which to compute features, one of 'day', 'week', or
        'month'.

    target_column : str
        Name of column to use for target computation.

    target_value : str, default None
        If `target_column` is categorical, the value in `target_column` to count to compute
        the target.

    events : list of str, default None
        List of values `event_col` column whose counts to turn into features. If None,
        simply use the `histogram_top_k` most frequent values of `event_col`.

    histogram_top_k : int, default 50
        For `event_col` column and `histogram_cols` columns with no explicit value list,
        the counts of the `histogram_top_k` most frequent values are turned into features.

    histogram_cols : list of str or tuple, default None
        Names of columns whose value counts to turn into features. List item can also be a
        tuple with (column name, whitelist of values whose counts to turn into features)
        if the automatic behavior of using the `histogram_top_k` most frequent values is not
        desired.

    top_k : int, default 10
        Number of distinct values of categorical columns (such as mode) in the feature table
        to retain. All other values are replaced with a placeholder like '_OTHER_'.

    drop_null : bool, default True
        Drop rows in the feature table with null features.

    n_time_periods_per_row_for_features : int, default 1
        Number of previous periods' features to include in the current row.

    n_time_periods_per_row_for_target : int, default 1
        Number of future periods' `target_column` values or value counts to sum to get
        the target for the current row.

    filter_inactive : bool, default False
        Only return rows in the feature and inference tables where target is nonzero
        for at least one of the current period and last `n_time_periods_per_row_for_features`.

    binarize_target : bool, default False
        Binarize the target variable by setting it to 1 if its value is greater than
        `binarize_threshold`, and 0 otherwise.

    binarize_threshold : int, default 0
        Threshold for target binarization.

    Returns
    _______
    Feature table (pyarrow.Table), inference table (pyarrow.Table), and metadata (dict) about
    each feature (used by training and inference). The inference table
    contains the rows for the most recent time period in the event table, which can be used
    to make forward-looking predictions.
    """

    if histogram_cols == None:
        histogram_cols = []

    # Roll up important cols/fields into one tuple
    fields = (event_col, join_field, time_field)

    # Do some initial booking to track col types
    num_cols = []
    cat_cols = []
    for field in wrangled_table.schema:
        if field.name == join_field:
            continue
        if field.type in categorical_types:
            cat_cols.append(field.name)
        if field.type in numeric_types:
            num_cols.append(field.name)

    # Add duckdb connection for fast in-memory SQL
    con = duckdb.connect(database=':memory:')

    # Tally up the total number of time windows and cast time to microsecs 
    wrangled_table, tot_num_time_periods, start_time = _process_tot_num_time_periods(
        wrangled_table, time_field, time_period)
    wrangled_table = wrangled_table.set_column(
        wrangled_table.schema.get_field_index(time_field), time_field,
        pc.cast(wrangled_table[time_field], pa.timestamp('us')))

    # Put in dummy events for correct time splitting behavior
    dummy_events = _create_dummy_events(con, wrangled_table,
        join_field, time_field, time_period, start_time.strftime("%Y-%m-%d"),
        tot_num_time_periods)
    wrangled_table = pa.concat_tables([wrangled_table, dummy_events],
        promote=True)

    # Process heavy cols
    histogram_cols = map(lambda c: _clean_name(c) if '->' in c else c, histogram_cols)
    heavy_cat_spec = [(event_col, events)]
    for h in histogram_cols:
        heavy_cat_spec.append(h if type(h) == tuple else (h, None))
    heavy_cat_cols = []
    for col, vals in heavy_cat_spec:
        if vals == None: # User has not provided value whitelist
            p_vals = _process_top_k(histogram_top_k if col == event_col else top_k,
                col, wrangled_table, con)
            vals = [str(val) for val in p_vals[col] if val.as_py() != None]
        heavy_cat_cols.append((col, vals))

    # Core feature computation routine 
    feature_table = _compute_features(con, wrangled_table,
        join_field, time_field, heavy_cat_cols, cat_cols, num_cols,
        n_time_periods_per_row_for_features)

    # Apply top k to categorical (str) feature cols
    for S in feature_table.schema:
        if S.type == pa.string() and S.name != join_field:
            vals = _process_top_k(top_k, S.name, feature_table, con)
            vals = [str(val) for val in vals[S.name]]
            filtered_col = _process_cat_value_filter(
                vals, S.name, feature_table, con)
            idx = feature_table.column_names.index(S.name)
            feature_table = feature_table.set_column(
                idx, S.name, filtered_col[S.name])

    # Compute the target column
    labels = _compute_labels(con, feature_table, join_field, target_column, [target_value],
        n_time_periods_per_row_for_target)

    # Merge feature and target columns
    feature_table = con.execute(f"""SELECT feature_table.*, labels.label FROM feature_table JOIN labels ON
                          feature_table.{join_field} = labels.{join_field} AND feature_table.time_period = labels.time_period
                       """).arrow()
    if filter_inactive:
        feature_table = _filter_inactive(feature_table, event_col, target_value,
            n_time_periods_per_row_for_features=n_time_periods_per_row_for_features)

    # Compute inference table
    max_date = pc.max(feature_table['time_period']).as_py().strftime("%Y-%m-%d")
    inf_table = con.execute(
        f"SELECT * from feature_table WHERE time_period = '{max_date}'").arrow()

    if binarize_target:
        new_label = pc.greater(feature_table['label'], binarize_threshold)
        feature_table = feature_table.drop(['label']).append_column('label', new_label)

    # Drop missing labels in feature_table (those are inference points)
    if drop_null:
        feature_table = con.execute(
            f"""
                SELECT * FROM feature_table 
                WHERE feature_table.label IS NOT NULL;
            """).arrow()

    return (feature_table, inf_table,
            _compute_metadata(feature_table, fields, con))

def _filter_inactive(feature_table, event_col, target_value, min_active_time_periods=1,
    n_time_periods_per_row_for_features=1):

    col_names = _gen_col_names(event_col, [target_value])
    time_periods = []
    for i in range(n_time_periods_per_row_for_features + 1):
        suffix = "" if i == 0 else f"_{i}"
        agg = ' + '.join([f"COALESCE({c}_count{suffix}, 0)" for c in col_names])
        time_periods.append(f"(CASE WHEN {agg} > 0 THEN 1 ELSE 0 END)")
    time_periods = ' + '.join(time_periods)
    con = duckdb.connect(database=':memory:')
    return con.execute(f"SELECT * FROM feature_table WHERE {time_periods} >= {min_active_time_periods}").arrow()

def _gen_col_names(col_name, val_list):
    # TODO: Rename this and clean up naming notation
    return list(map(lambda e: col_name + '_' + _clean_name(e), val_list))

def _process_tot_num_time_periods(wrangled_table, time_field, time_period):
    floored_dates = pc.floor_temporal(
        wrangled_table[time_field], unit=time_period)
    wrangled_table = wrangled_table.set_column(
        wrangled_table.schema.get_field_index(time_field),
        time_field, floored_dates)
    start_day = pc.min(floored_dates).as_py()
    end_day = pc.max(floored_dates).as_py()
    if time_period == 'week':
        num_time_periods = (end_day - start_day).days // 7
    elif time_period == 'month':
        num_time_periods = (end_day.year - start_day.year) * 12
        num_time_periods += (end_day.month - start_day.month)
    else: # feature time_period == day
        num_time_periods = (end_day - start_day).days
    num_time_periods += 1 # end date is inclusive
    return wrangled_table, num_time_periods, start_day

def _compute_metadata(feature_table, cols, con):
    metadata = dict()
    metadata['fields'] = dict()
    metadata['event_col'] = cols[0]
    metadata['join_field'] = cols[1]
    metadata['time_field'] = 'time_period'
    m = metadata['fields']
    for col in feature_table.schema:
        m[col.name] = {'dtype' : str(col.type)}
        if _is_numeric(str(col.type)):
            min_max = pc.min_max(feature_table[col.name])
            m[col.name]['min'] = min_max['min'].as_py()
            m[col.name]['max'] = min_max['max'].as_py()
        else: 
            distincts = con.execute(
                f"SELECT DISTINCT {col.name} FROM feature_table").arrow()
            distincts_py = [val.as_py() for val in distincts[col.name]]
            m[col.name]['distincts'] = distincts_py
    return metadata

def _compute_labels(con, feature_table, join_field, label_col, label_values,
    n_time_periods_per_row_for_target):
    if len(label_values) == 1 and label_values[0] == None:
        col_names = [f"{label_col}_sum"]
    else:
        col_names = map(lambda c: f"{c}_count", _gen_col_names(label_col, label_values))
    expr = ' + '.join(map(lambda c: ' + '.join([f"(LEAD({c}, {i}) OVER win)" \
        for i in range(1, n_time_periods_per_row_for_target + 1)]), col_names))
    return con.execute(f"""
                            SELECT {join_field}, time_period, {expr} AS label 
                            FROM feature_table 
                            WINDOW win AS
                            (PARTITION BY {join_field} ORDER BY time_period ASC)
                        """).arrow()

def _compute_features(con, event_table, join_field, time_field,
    heavy_cat_cols, cat_cols, num_cols, preceding_time_periods):

    feat_names = []

    # Run value-enumeration for heavy cat cols
    cases = []
    for col, vals in heavy_cat_cols:
        col_names = _gen_col_names(col, vals)
        for e_sql, v_sql in zip(vals, col_names):
            feat_name_sql = f"{v_sql}_count"
            cases.append(f"""
                            CAST(SUM(CASE WHEN {col} = '{e_sql}'
                            THEN 1 ELSE 0 END) AS int64) AS {feat_name_sql}
                        """)
            feat_names.append(feat_name_sql)

    # Run lightweight categorical aggregations for cat columns
    cats = []
    for cat in cat_cols:
        cats.append(f"COUNT(DISTINCT {cat}) AS {cat}_count_distinct")
        cats.append(f"CAST(MODE({cat}) AS VARCHAR) AS {cat}_mode")
        feat_names.extend([f"{cat}_count_distinct", f"{cat}_mode"])

    # Run lightweight numeric aggregations for numeric columns
    conts = []
    for cont in num_cols:
        conts.append(f"CAST(SUM({cont}) AS double) AS {cont}_sum")
        conts.append(f"CAST(AVG({cont}) AS double) AS {cont}_avg")
        conts.append(f"CAST(STDDEV({cont}) AS double) AS {cont}_std")
        feat_names.extend([f"{cont}_sum", f"{cont}_avg", f"{cont}_std"])
    base_feats = ', '.join(cases + cats + conts)
    base_features = con.execute(f"""SELECT {join_field} AS {join_field}, {time_field} AS time_period,
                                    {base_feats} FROM event_table GROUP BY 1, 2;
                                """).arrow()
    final_feats = []
    for feat_name in feat_names:
        lags = ', '.join([f"LAG({feat_name}, {i}) OVER win AS {feat_name}_{i}" \
            for i in range(1, preceding_time_periods + 1)])
        final_feats.append(f"{feat_name}, {lags}")
    final_feats = ', '.join(final_feats)
    return con.execute(f"""SELECT {join_field}, time_period, {final_feats} FROM base_features WINDOW win AS
                        (PARTITION BY {join_field} ORDER BY time_period ASC);
                        """).arrow()


def _create_dummy_events(con, event_table, join_field, time_field,
    feature_time_period, start_day, num_time_periods):

    users = con.execute(f"""SELECT DISTINCT {join_field} AS {join_field}
                            FROM event_table
                         """).arrow()
    iters = pa.table({'iter': pa.array([i for i in range(num_time_periods)])})
    if feature_time_period == 'day':
        interval = "iter day"
    elif feature_time_period == 'week':
        interval = "(7 * iter) day"
    else: # 'month'
        interval = "iter month"
    dates = con.execute(f"""SELECT CAST(date_trunc('{feature_time_period}', DATE '{start_day}') +
                            INTERVAL {interval} AS timestamp) AS {time_field} FROM iters
                         """).arrow()
    return con.execute(f"""SELECT {join_field}, {time_field}
                           FROM users FULL JOIN dates ON 1 = 1
                        """).arrow()

def _process_top_k(top_k, field, _table, con):
    return con.execute(
        f"""
            SELECT {field}, COUNT({field}) AS cnt
            FROM _table
            GROUP BY {field}
            ORDER BY cnt DESC
            LIMIT {top_k};
        """).arrow()

def _process_cat_value_filter(top_k_vals, field, _table, con):
    top_k_vals_sql = ""
    for i,val in enumerate(top_k_vals):
        top_k_vals_sql += f"{field} = '{val}' "
        if i < len(top_k_vals) - 1:
            top_k_vals_sql += "OR "

    other = "__OTHER__"
    while other in top_k_vals:
        other += "_"
    other_sql = f"'{other}'"
    
    return con.execute(
        f"""
            SELECT CASE 
            WHEN {top_k_vals_sql} THEN {field} ELSE {other_sql} END
            AS {field} FROM _table;
        """).arrow()
