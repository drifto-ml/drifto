import duckdb
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

def wrangle(
    join_field,
    time_field,
    primary_table_path=None,
    cols=None,
    event_col=None,
    table_paths=None,
    start_time=None,
    end_time=None):
    """
    Combine multiple event tables into a merged event table.

    Parameters
    __________
    join_field : str
        Name of column containing entitites to compute features over (e.g. user ID),
        must be present in all event tables.

    time_field : str
        Name of timestamp column, must be present in all event tables.

    primary_table_path : str, default None
        Path to Parquet file with the primary event table, which must also contain
        a column (`event_col`) with the type of each event. If None, attempt to use the
        first path in `table_paths`.

    cols : list of str, default None
        List of column names besides `join_field`, `time_field`, and `event_col` (for the
        primary table) to keep in the merged event table. Column names must be unique across
        the event tables.

    event_col : str, default None
        Name of the column with the event type in the primary event table. If None, will take
        the first item in `cols`.

    table_paths : list of tuple of str, str, default None
        List of tuples of (event type to give to all events in this table, path to Parquet
        file with event table).

    start_time : datetime, default None
        Filter for events with timestamp greater than this.

    end_time : datetime, default None
        Filter for events with timestamp less than this.

    Returns
    _______
    Merged event table (pyarrow.Table).
    """

    if cols == None:
        cols = []
    if table_paths == None:
        table_paths = []

    # We require an event col
    assert event_col != None or len(cols) >= 1, "We require an event col"

    # Process the event col
    print("Drifto.Wrangle: Processing event col...")
    if event_col == None:
        event_col = cols[0]
    else:
        cols.insert(0, event_col)

    # Process the table paths
    print("Drifto.Wrangle: Processing table paths...")
    if primary_table_path == None:
        primary_table_path = table_paths[0]
    else:
        table_paths.insert(0, ("primary_table", primary_table_path))

    join_schemas = list(map(lambda p: _read_schema(p[1]), table_paths))

    # Check core assertions
    print("Drifto.Wrangle: Checking core assertions...") 
    assert event_col in join_schemas[0].names, "Event col not found in primary table"
    for i, ((_, path), js) in enumerate(zip(table_paths, join_schemas)):
        assert join_field in js.names, f"Join field not found in {path}"
        assert time_field in js.names, f"Time field not found in {path}"
        if i > 0:
            assert not event_col in js.names, f"Event col found in {path}"

    cols_to_read = [{join_field, time_field} for i in range(len(join_schemas))]
    cols_to_read[0].add(event_col)
    json_queries = []
    json_cols_to_remove = set()

    # Try to find additional cols
    print("Drifto.Wrangle: Processing columns...") 
    for column in cols:
        col = column.split('->')
        found = False
        for i, js in enumerate(join_schemas):
            if col[0] in js.names:
                if found:
                    warnings.warn(f'Column {col[0]} found in multiple tables.')
                cols_to_read[i].add(col[0])
                found = True
        if not found:
            warnings.warn(f'Column {col[0]} not found in any table.')
        if len(col) > 1: # JSON field
            json_queries.append(column)
            # We don't want any JSON-valued cols in our analysis
            json_cols_to_remove.add(col[0])
    
    tables = []
    filters = []
    if start_time:
        filters.append((time_field, '>=', start_time))
    if end_time:
        filters.append((time_field, '<=', end_time))
    if len(filters) == 0:
        filters = None
    for cols, (_, path) in zip(cols_to_read, table_paths):
        tables.append(pq.read_table(
            path,
            columns=list(cols),
            filters=filters))

    # Always assume join_field is categorical and therefore can cast to str
    for i in range(len(tables)):
        idx = tables[i].column_names.index(join_field)
        tables[i] = tables[i].set_column(
            idx, join_field, pc.cast(tables[i][join_field], pa.string()))
        if i > 0:
            # Add primary column to secondary tables...
            # Filled with just the table name
            tables[i] = tables[i].append_column(event_col,
                pa.nulls(len(tables[i]),
                    pa.string()).fill_null(table_paths[i][0]))

    full_table = pa.concat_tables(tables, promote=True)

    print("Drifto.Wrangle: Connecting to DuckDB...")
    con = duckdb.connect(database=':memory:')
    print("Drifto.Wrangle: Finalizing feature table...")
    for jq in json_queries:
        clean = _clean_name(jq)
        res = con.execute(f"SELECT {jq} FROM full_table").arrow()
        full_table = full_table.append_column(clean, res[0])
    full_table = full_table.drop(list(json_cols_to_remove))

    print("Drifto.Wrangle: Done!")
    return full_table

def _clean_name(orig_name):
    name = ''.join([i if i.isalnum() else '_' for i in orig_name])
    # remove duplicate underscores
    name = name.lstrip('_').rstrip('_')
    return ''.join([c for i, c in enumerate(name) if c != '_' or name[i - 1] != '_'])

def _read_schema_remote(path):
    s3 = pa.fs.S3FileSystem()
    return pq.ParquetDataset(path, filesystem=s3).schema

def _read_schema(path):
    if path[:5] == 's3://':
        return _read_schema_remote(path[5:])
    else:
        return pq.read_schema(path)
