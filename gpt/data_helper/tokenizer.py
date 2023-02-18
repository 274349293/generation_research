# -*-coding:utf-8-*-
import hashlib
import sqlalchemy
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import pymysql
from pydantic import BaseModel
from tqdm import tqdm


def mysqldb():
    db = pymysql.connect(host="172.29.28.66", user="root", password="123456", database="generation_research", port=3306,
                         autocommit=False)
    return db


class AdditionalSpecialTokens(BaseModel):
    # word embedding
    space_token = "[Space]"

    # task token
    tag_task_token = "[TASK:TAG]"

    # segment embedding
    tag_token = '[Tag]'
    content_token = '[Content]'
    img_token = "[IMG]"
    speaker1_token = '[Speaker1]'
    speaker2_token = '[Speaker2]'
    task1_token = '[TASK1]'
    task2_token = '[TASK2]'
    task3_token = '[TASK3]'


class DataTokenizer:
    def __init__(self, logger):
        self.logger = logger
        self.db = mysqldb()
        self.cleaning_data = self.get_cleaning_data()
        self.vocab_path = '/home/liulei/PycharmProjects/generation_research/gpt/utils/vocab.txt'
        self.n_ctx = 1020
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_path, do_lower_case=True, max_len=self.n_ctx)
        self.special_tokens_obj = AdditionalSpecialTokens()
        self.special_tokens = self.special_tokens_obj.__dict__.values()
        self.tokenizer.add_special_tokens({'additional_special_tokens': [x for x in self.special_tokens]})
        self.content_seg_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens_obj.content_token)
        self.df_token_data = self.get_df_token_data()

    def get_cleaning_data(self):
        return np.array(pd.read_sql("SELECT * FROM t_case_cleaning", self.db))

    def tokenize_one_sample(self, sample):
        """
        数据处理函数
        Args:
            sample: str:'content'
        Returns:
        """

        input_ids = []
        token_type_ids = []

        # 对文本进行tokenizer.tokenize分词
        content_tokens = self.tokenizer.tokenize(sample)

        # 生成模型所需的input_ids和token_type_ids
        '''cls'''
        input_ids.append(self.tokenizer.cls_token_id)

        '''content'''
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))

        '''sep'''
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.extend([self.content_seg_id] * len(input_ids))

        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)

        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.n_ctx, f"{len(input_ids)}"

        return input_ids, token_type_ids

    def get_df_token_data(self):
        token_data = []
        for item in tqdm(self.cleaning_data, desc='data tokenizer'):
            input_ids, token_type_ids = self.tokenize_one_sample(item[3])
            token_data.append([item[1], item[2], str(input_ids), str(token_type_ids)])
        return pd.DataFrame(token_data, columns=['case_id', 'disease', 'word_embedding', 'segment_embedding'])

    def insert_case_data(self):
        self.df_token_data.to_sql(name='t_case_token', con=sqlalchemy.create_engine(
            "mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4".format(
                'root', '123456', '172.29.28.66', 3306, 'generation_research')), if_exists='fail')
