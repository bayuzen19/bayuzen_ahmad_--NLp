import pandas as pd
from pathlib import Path
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys

class DataIngestion:
    """
    Kelas untuk melakukan ingestion data dari file CSV.

    Attributes:
        data_path (str): Path menuju file CSV yang akan di-ingest.

    Methods:
        initiate_data_ingestion() -> pd.DataFrame:
            Memulai proses ingestion data dari file CSV.
            Menghapus kolom "Unnamed: 0" dan baris dengan nilai kosong pada kolom "Review Text".
            Mengembalikan DataFrame yang telah diproses.

    Raises:
        CustomException: Jika terjadi exception selama proses ingestion data.
    """


    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Started data ingestion")
            df = pd.read_csv(self.data_path)
            df = df.drop("Unnamed: 0", axis=1)
            df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)