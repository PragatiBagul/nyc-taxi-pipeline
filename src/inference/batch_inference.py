"""
Batch Inference script that :
- reads incoming trips(parquet/csv)
- joins with feature store (Duck DB)
- loads an MLflow model (models:/ or runs:/)
- produces predictions and writes them to parquet
"""
import argparse
import duckdb
import mlflow.pyfunc
import os
import pandas as pd

def join_input_with_features(input_path:str,feature_path:str,join_on:str='pickup_date') -> pd.DataFrame:
    """
    Join input events with a feature store parquet on a date bucket column (pickup_date)
    Assumes both input and feature store contain pickup_date (string YYYY-MM-DD)
    """
    con = duckdb.connect()
    #support both csv/parquet inputs by DuckDB helper
    input_read =  f"read_parquet('{input_path}')" if input_path.endswith('parquet') else f"read_csv_auto('{input_path}')"
    sql = f"""
    SELECT i.*. f.*
    FROM {input_read} i
    LEFT JOIN read_parquet('{feature_path}/*.parquet') f
    on i.{join_on} = f.{join_on}
    """
    return con.execute(sql).fetchdf()

def load_model(model_uti:str):
    """
    Load model using MLFlow PyFunc API.
    """
    return mlflow.pyfunc.load_model(model_uti)

def run_batch(input_path:str,feature_path:str,model_uri:str,output_path:str,batch_limit:int=None):
    df = join_input_with_features(input_path,feature_path)
    if batch_limit:
        df = df.head(batch_limti)
    model = load_model(model_uri)

    #Determine feature columns expected by model if possible -> here we assume model accepts numeric columns present
    X = df.select_dtypes(include=['number'])
    preds = model.predict(X)
    df['pred_trip_duration'] = preds

    os.makedirs(output_path,exist_ok=True)
    out_file = os.path.join(output_path,'predictions.parquet')
    df.to_parquet(out_file,index=False)
    print(f'Wrote predictions to {out_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True,help='Input parquet file')
    parser.add_argument('--feature-path',required=True,help='Feature store parquet file')
    parser.add_argument('--model-uri',required=True,help='MLflow model URI (models:/ or runs:/')
    parser.add_argument('--output',required=True,help='Output folder to write predictions')
    parser.add_argument('--limit',type=int,default=None,help='Optional (limit rows: processed for dev)')
    args = parser.parse_args()
    run_batch(args.input,args.feature_path,args.model_uri,args.output,args.limit)