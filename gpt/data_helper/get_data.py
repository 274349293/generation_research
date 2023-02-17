import pymysql
import numpy as np
import sqlalchemy
import pandas as pd
from tqdm import tqdm
import torch
import os


def mysqldb():
    db = pymysql.connect(host="172.29.28.66", user="root", password="123456", database="dw", port=3306,
                         autocommit=False)
    return db


class PretrainData:
    def __init__(self, logger):
        self.logger = logger
        self.db = mysqldb()
        self.t_case_data = self.get_case_df()

    def get_case_df(self):
        return pd.read_sql("SELECT case_id, disease, txt FROM t_case_with_tcm", self.db)

    def insert_case_data(self):
        self.t_case_data.to_sql(name='t_case_origin', con=sqlalchemy.create_engine(
            "mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4".format(
                'root', '123456', '172.29.28.66', 3306, 'generation_research')), if_exists='fail')
