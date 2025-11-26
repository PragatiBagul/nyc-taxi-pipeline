"""
Simple Parquet based feature store using DuckDB for fast querying
"""
import os
import duckdb
import pandas as pd
from typing import Optional

def write_parquet(df: pd.DataFrame,path:str,partition_cols=None,overwrite=False):
    if overwrite and os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    df.to_parquet(path,partition_cols=partition_cols,engine='pyarrow',index=False)

def query_parquet(parquet_glob:str,where:Optional[str]=None,limit:Optional[int]=None):
    con = duckdb.connect()
    sql = f"SELECT * FROM read_parquet('{parquet_glob}')"

    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {limit}"
    return con.execute(sql)

