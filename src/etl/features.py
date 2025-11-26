"""
Feature engineering for NYC Taxi Pipeline
Creates distance, direction, temporal, and speed features
"""
import numpy as np
import pandas as pd

EARTH_RADIUS = 6371

def haversine_distance(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance calculation"""
    lon1,lat1,lon2,lat2 = map(np.radians, [lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))

def add_haversine_distance(df:pd.DataFrame):
    df['haversine_distance'] = haversine_distance(df['lon'], df['lat'], df['lon'], df['lat'])
    return df

def manhattan_distance(lon1, lat1, lon2, lat2):
    """Vectorized manhattan distance calculation"""
    # Distance when moving only vertically (north/south)
    a = haversine_distance(lat1, lon1, lat2, lon1)
    # Distance when moving only horizontally (east/west)
    b = haversine_distance(lat1, lon1, lat1, lon2)
    return a + b

def add_manhattan_distance(df:pd.DataFrame):
    df['manhattan_km'] = manhattan_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )
    return df

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(x, y))
    return bearing

def add_bearing(df:pd.DataFrame):
    df['bearing'] = calculate_bearing(df['pickup_latitude'],
                                      df['pickup_longitude'],
                                      df['dropoff_latitude'],
                                      df['dropoff_longitude']
                                      )
    return df

def add_time_features(df:pd.DataFrame) -> pd.DataFrame:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Extract features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour.astype('category')
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek.astype('category')
    df['pickup_month'] = df['pickup_datetime'].dt.month.astype('category')
    df['pickup_date'] = df['pickup_datetime'].dt.date

    df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int).astype('category')

    return df

def add_speed_features(df:pd.DataFrame) -> pd.DataFrame:
    # Convert duration from seconds to hours
    df['duration_hours'] = df['trip_duration'] / 3600
    # Average speed in km/h
    df['average_speed'] = df['haversine_km'] / df['duration_hours']
    # Remove speeds > 200 km/h (impossible for taxis)
    df = df[df['average_speed'] < 200]
    # Remove speeds < 1 km/h (likely idle / data issues)
    df = df[df['average_speed'] > 1]
    return df

def add_all_features(df:pd.DataFrame) -> pd.DataFrame:
    df = add_haversine_distance(df)
    df = add_manhattan_distance(df)
    df = add_bearing(df)
    df = add_time_features(df)
    df = add_speed_features(df)
    return df