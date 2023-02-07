import pymysql
from tqdm import tqdm
import torch
import os


class OriginData:
    """
    get origin data from db

    """
    db = pymysql.connect(host="172.30.2.221", user="root",
                         password="123456", database="dw", port=3306, autocommit=False)

    sql = 'select * from t_case_with_tcm'
    cursor = db.cursor()
    row_count = cursor.execute(sql)


class PretrainData:
    def __init__(self, logger, single_cache_size, file_dir, use_mysql_cache):
        self.db = OriginData()
        self.single_cache_size = single_cache_size
        self.file_dir = file_dir
        self.logger = logger
        self.use_mysql_cache = use_mysql_cache

    def get_mysql_data(self):
        mysql_data = []
        nlp_data_files = []
        file_number = 0
        for line in tqdm(self.db.cursor.fetchall(), desc="load mysql data"):
            disease_name = line[1]
            subtype = line[2]
            matching_level = line[3]
            keyword_dict = line[4]
            content = line[5]
            case_info = line[6]
            seged = line[10]
            new_wd = line[11]

            mysql_data.append([disease_name, subtype, matching_level, keyword_dict, content, case_info, seged, new_wd])
            # 分片保存
            if len(mysql_data) == self.single_cache_size:
                fn = f'mysql_data_{str(file_number)}_size_{len(mysql_data)}'
                f = self.file_dir + fn
                self.logger.info(f"caching mysql data in: {f}")
                torch.save(mysql_data, f)
                nlp_data_files.append(f)
                mysql_data = []
                file_number += 1

        if len(mysql_data):
            fn = f'mysql_data_{str(file_number)}_size_{len(mysql_data)}'
            f = self.file_dir + fn
            self.logger.info(f"caching mysql data in: {f}")
            torch.save(mysql_data, f)
            nlp_data_files.append(f)

        return nlp_data_files

    def load_data(self):
        if self.use_mysql_cache:
            self.logger.info(f"using mysql data cache")
            return [self.file_dir + file for file in os.listdir(self.file_dir)]
        else:
            self.logger.info(f"not using mysql data cache")
            return self.get_mysql_data()
