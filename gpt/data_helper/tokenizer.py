# -*-coding:utf-8-*-
import hashlib
from concurrent.futures import as_completed, ProcessPoolExecutor
from torchvision import transforms
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import pymysql
import torch
from tqdm import tqdm
import os


def mysqldb():
    db = pymysql.connect(host="172.29.28.66", user="root", password="123456", database="generation_research", port=3306,
                         autocommit=False)
    return db


class AdditionalSpecialTokens:
    # word embedding
    space_token = "[Space]"

    # task token
    tag_task_token = "[TASK:TAG]"

    # segment embedding
    comment_token = "[Comment]"
    tag_token = '[Tag]'
    question_token = '[Question]'
    answer_token = '[Answer]'
    content_token = '[Content]'
    img_token = "[IMG At Here]"
    task_token = '[TASK]'
    speaker1_token = '[Speaker1]'
    speaker2_token = '[Speaker2]'
    comment1_token = "[Comment1]"
    comment2_token = "[Comment2]"
    comment3_token = "[Comment3]"
    comment4_token = "[Comment4]"
    comment5_token = "[Comment5]"
    comment6_token = "[Comment6]"
    comment7_token = "[Comment7]"
    comment8_token = "[Comment8]"
    comment9_token = "[Comment9]"
    comment10_token = "[Comment10]"
    article_start_token = "[ArticleStart]"
    article_end_token = "[ArticleEnd]"
    article_split_token = "[ArticleSplit]"


class DataTokenizer:
    def __init__(self, logger):
        self.logger = logger
        self.db = mysqldb()
        self.df_cleaning_data = self.get_cleaning_data()
        self.vocab_path = '/home/liulei/PycharmProjects/generation_research/gpt/utils/vocab.txt'
        self.n_ctx = 1020
        self.tokenizer = BertTokenizer(vocab_file=self.vocab_path, do_lower_case=True, max_len=self.n_ctx)
        self.special_tokens_obj = AdditionalSpecialTokens()
        self.special_tokens = self.special_tokens_obj.__dict__.values()
        self.tokenizer.add_special_tokens({'additional_special_tokens': [x for x in self.special_tokens]})

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
        print(content_tokens)

        # 生成模型所需的input_ids和token_type_ids
        '''cls'''
        input_ids.append(self.tokenizer.cls_token_id)
        print(input_ids)
        '''content'''
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
        print(input_ids)
        '''sep'''
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.extend([self.content_seg_id] * len(input_ids))

        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)

        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.n_ctx, f"{len(input_ids)}"

        return input_ids, token_type_ids

    def get_tokens(self):
        for item in self.df_cleaning_data:
            token_sample = self.tokenize_one_sample(item[3])
            exit()
