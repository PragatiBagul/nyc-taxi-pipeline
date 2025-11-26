"""
Small loader utilities for the NYC dataset supporting both pandas and dask
"""
from typing import Optional
import pandas as pd

def load_pandas(path:str,
                parse_dates: list = ['pickup_datetime','dropoff_datetime'],
                nrows: Optional[int]=None) -> pd.DataFrame:
    """
    Load CSV using Pandas
    Args:
    path (str): Path to CSV file
    parse_dates (Optional[list]): List of column names to use to parse dates
    nrows (Optional[int]): Number of rows to read
    Returns:
    pd.DataFrame: Pandas DataFrame
    """
    return pd.read_csv(path, parse_dates = parse_dates, nrows = nrows)

def load_dask(path:str,
              parse_dates:list=['pickup_datetime','dropoff_datetime'],
              npartitions:int=32) -> pd.DataFrame:
    """
    Load CSV into a Dask DatFrame and repartition.
    Requires dask to be installed.
    Returns a dask.dataframe.DataFrame
    """
    import dask.dataframe as dd
    df = dd.read_csv(path,assume_missing=True,parse_dates=parse_dates)
    return df.repartition(npartitions=npartitions)

def to_parquet(df,path:str,partition_cols:Optional[list]=None,overwrite:bool=False) -> pd.DataFrame:
    """
    Persist pandas or dask DataFrame to Parquet format (pyarrow engine)
    If dask is Dask, this delegates to df.to_parquet(); otherwise uses pandas.to_parquet()
    """
    if overwrite:
        import os, shutil
        if os.path.exists(path):
            shutil.rmtree(path)

    #Dask DataFrame
    try:
        import dask.dataframe as dd
        if isinstance(df,dd.DataFrame):
            df.to_parquet(path,partition_cols=partition_cols,engine="pyarrow",write_index=False)
            return
    except Exception:
        pass

    df.to_parquet(path,partition_cols=partition_cols,engine="pyarrow",index=False)

if __name__ == 'main':
    import argparse
    parser = argparse.ArgumentParser(description='Simple CSV -> Parquet Converter')
    parser.add_argument('--csv',required=True,help='input CSV path')
    parser.add_argument('--out',required=True,help='output parquet directory')
    parser.add_argument('--partitions',type=int,default=8,help='npartitions when using dask')
    parser.add_argument('--use-dask',action='store_true',help='use dask to read and write')
    args = parser.parse_args()

    if args.use_dask:
        df = load_dask(args.csv,npartitions=args.partitions)
    else:
        df = load_pandas(args.csv)
    to_parquet(df,args.out,partition_cols=['pickup_date'] if 'pickup_date' in (df.columns if hasattr(df,'columns') else []) else None,overwrite=True)
    print('Wrote parquet to ',args.out)