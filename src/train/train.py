"""
Train a regression model using features stored as Parquet (DuckDB). Logs experiments to MLFlow
Example :
    python -m src.train.train --feature_path 'feature_path/trip_features' --start '2015-01-01' --end '2015-06-30'
"""

import argparse
import mlflow
import mlflow.sklearn
import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_features_duckdb(parquet_glob:str,where:str=None) -> pd.DataFrame:
    con = duckdb.connect()
    sql = f"SELECT * FROM read_parquet('{parquet_glob}/*.parquet')"
    if where:
        sql += f" WHERE {where}"
    return con.execute(sql).fetchdf()

def prepare_data(df:pd.DataFrame,label_col:str = 'trip_duration',time_col:str='pickup_datetime') -> pd.DataFrame:
    # basic dropna and ordering by time for time-based split
    df = df.dropna(subset=[label_col])
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col],errors='coerce')
        df = df.sort_values(by=[time_col])
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X,y

def train_lightgbm(X_train,y_train,X_val,y_val,params):
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError('LightGBM not installed') from e

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=50,verbose=False)
    return model

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main(args):
    where = None
    if args.start and args.end:
        where = f"pickup_date BETWEEN '{args.start}' AND '{args.end}'"
    df = load_features_duckdb(args.feature_path,where)
    print('loaded features')

    X,y = prepare_data(df,label_col = args.label_col,time_col = 'pickup_datetime')

    if args.time_based:
        split_idx = int(len(X)*(1-args.val_fraction))
        X_train, X_val = X.iloc[:split_idx],X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=args.val_fraction,random_state=42)

    # Select numeric columns only for training
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    X_train_nums = X_train[numeric_cols]
    X_val_nums = X_val[numeric_cols]

    #MLFlow Experiment
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_param('model','LightGBM')
        mlflow.log_param('nrows',len(X))
        mlflow.log_param('num_features',len(numeric_cols))

        params = {'n_estimators': args.n_estimators,'learning_rate':args.learning_rate,'num_leaves':args.num_leaves}
        mlflow.log_params(params)

        model = train_lightgbm(X_train_nums,y_train,X_val_nums,y_val,params)

        preds = model.predict(X_val_nums)
        score = rmse(y_val,preds)
        mlflow.log_metric('rmse',score)

        mlflow.sklearn.log_model(model,'model',registered_model_name=args.register_name if args.register else None)
        print(f'Run finished. RMSE= {score}. Run ID= {run.info.run_id}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-path',required=True,help='Path to parquet feature folder (glob parent)')
    parser.add_argument('--start',default=None,help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end',default=None,help='End date (YYYY-MM-DD)')
    parser.add_argument('--label-col',default='trip_duration',help='Label column name')
    parser.add_argument('--experiment-name',default='nyc-taxi-duration')
    parser.add_argument('--time-based',action='store_true',help='Use time based split instead of random')
    parser.add_argument('--val-fraction',type=float,default=0.2,help='Fraction of validation set')
    parser.add_argument('--n-estimators',dest='n_estimators',type=int,default=1000)
    parser.add_argument('--learning-rate',dest='learning_rate',type=float,default=0.1)
    parser.add_argument('--num-leaves',dest='num_leaves',type=int,default=31)
    parser.add_argument('--register',action='store_true',help='Register model in Mlflow Model registry')
    parser.add_argument('--register-name',default='nyc-taxi-model')
    args = parser.parse_args()
    main(args)