from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df

@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path = data_path)
    clean_df(df)