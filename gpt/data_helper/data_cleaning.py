import pymysql
import numpy as np
import sqlalchemy
import pandas as pd
from tqdm import tqdm
import torch
import os


def mysqldb():
    db = pymysql.connect(host="172.29.28.66", user="root", password="123456", database="generation_research", port=3306,
                         autocommit=False)
    return db


class DataCleaning:
    def __init__(self, logger):
        self.logger = logger
        self.db = mysqldb()
        self.t_case_origin_data = self.get_case_origin_df()
        self.df_cleaning_data = self.data_cleaning()

    def get_case_origin_df(self):
        return np.array(pd.read_sql("SELECT * FROM t_case_origin", self.db))

    def data_cleaning(self):
        cleaning_data = []
        for item in self.t_case_origin_data:
            case_content = item[3].replace('初步诊断:', '').replace(' ', '')
            case_content = case_content[0:1000] if len(case_content) > 1000 else case_content
            cleaning_data.append([item[1], item[2], case_content])

        return pd.DataFrame(cleaning_data, columns=['case_id', 'disease', 'txt'])

    def insert_case_data(self):
        self.df_cleaning_data.to_sql(name='t_case_cleaning', con=sqlalchemy.create_engine(
            "mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4".format(
                'root', '123456', '172.29.28.66', 3306, 'generation_research')), if_exists='fail')
