"""
src/etl/clean.py
Data cleaning+quality checks for NYC Taxi Trip Dataset.
Designed to work with Pandas Dataframe (Dask via map_partitions).
"""
import pandas as pd
import numpy as np
from typing import Tuple

from pandas import DataFrame


def find_outliers(df: pd.DataFrame,TARGET:str = 'trip_duration') -> tuple[DataFrame, float, float]:
    Q1 = df[TARGET].quantile(0.25)
    Q3 = df[TARGET].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = df[(df[TARGET] < lower_limit) | (df[TARGET] > upper_limit)]
    return outliers,lower_limit,upper_limit

def remove_outliers(
        df: pd.DataFrame,
        TARGET: str ='trip_duration'
) -> pd.DataFrame:
    outliers,lower_limit,upper_limit = find_outliers(df,TARGET)
    print(f'Outliers found: {len(outliers)} , percentage : {len(outliers)/len(df)*100}%')
    # Calculating the percentage of outliers higher than the upper limit
    higher_outlier = len(df[df[TARGET] > upper_limit]) / len(df[TARGET])
    print(f'Percentage of outliers greater than higher limit found: {higher_outlier*100}%')
    lower_outlier = len(df[df[TARGET] < lower_limit]) / len(df[TARGET])
    print(f'Percentage of outliers lesser than lower limit found: {lower_outlier*100}%')
    #Drop outliers greater than the higher limit
    df = df[df[TARGET] < upper_limit]
    # Drop outliers lesser than the lower limit
    df = df[df[TARGET] > lower_limit]
    return df

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric summary statistics for a dataframe"""
    return df.describe()

def fill_missing_passenger_count(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing passenger count with 1"""
    if 'passenger_count' in df.columns:
        df['passenger_count'] = df['passenger_count'].fillna(1).astype(int)
    return df