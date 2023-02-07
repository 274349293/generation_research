from dzj_nlp.nlp_utils.logger_utils import TaskLogger
from dzj_nlp.nlp_data.get_data import Ner_Data
from dzj_nlp.data_helper.ner_data_helper import Ner_Data_helper
from dzj_nlp.tokenizer.ner_tokenizer import DataTokenizer
from dzj_nlp.data_loader.ner_dataloader import DataLoader
from dzj_nlp.nlp_configer import ner_configer as configer
import torch
from tqdm import tqdm


class DataModule:
    def __init__(self, hparams: configer.DataModuleParams):
        self.logger = hparams.logger
        self.single_cache_size = hparams.single_cache_size
        self.file_dir = hparams.file_dir
        self.use_mysql_cache = hparams.use_mysql_cache

    def load_data(self):
        return Ner_Data(self.logger, self.single_cache_size, self.file_dir, self.use_mysql_cache).load_data()


class DataHelperModule:
    def __init__(self, hparams: configer.DataHelperParams, nlp_data_files):
        self.logger = hparams.logger
        self.data_files_path = nlp_data_files
        self.file_dir = hparams.file_dir
        self.use_data_helper_cache = hparams.use_data_helper_cache
        self.window_size = hparams.window_size
        self.step_size = hparams.step_size

    def load_data(self):
        return Ner_Data_helper(self.logger, self.data_files_path, self.file_dir, self.use_data_helper_cache,
                               self.window_size, self.step_size).load_data()


class TokenizerModule:
    def __init__(self, hparams: configer.TokenizerModuleParams, data_helper_files):
        self.logger = hparams.logger
        self.data_helper_files = data_helper_files
        self.file_dir = hparams.file_dir
        self.tokenizer = hparams.tokenizer
        self.use_tokenizer_cache = hparams.use_tokenizer_cache
        self.worker_num = hparams.worker_num
        self.n_ctx = hparams.n_ctx
        self.target = hparams.target

        # # Special tokens
        # self.space_id = hparams.space_id
        # self.content_seg_id = hparams.content_seg_id
        # self.unk_id = hparams.unk_id
        # self.sep_id = hparams.sep_id

    def load_data(self):
        return DataTokenizer(self.logger, self.file_dir, self.data_helper_files, self.worker_num, self.n_ctx,
                             self.tokenizer, self.use_tokenizer_cache, self.target).load_data()


class DataLoaderModule:
    def __init__(self, hparams: configer.DataLoaderParams, tokens_files_path):
        self.logger = hparams.logger
        self.tokens_files_path = tokens_files_path

        self.device_type = hparams.device_type
        self.multi_gpu = hparams.multi_gpu
        self.random_seed = hparams.random_seed
        self.train_batch_size = hparams.train_batch_size
        self.train_sampler = hparams.train_sampler

        self.test_data_size = hparams.test_data_size
        self.test_batch_size = hparams.test_batch_size
        self.test_sampler = hparams.test_sampler
        self.data_loader_worker_num = hparams.data_loader_worker_num

    def get_data_loader_obj(self):
        return DataLoader(logger=self.logger, tokens_files_path=self.tokens_files_path, device_type=self.device_type,
                          multi_gpu=self.multi_gpu, random_seed=self.random_seed,
                          train_batch_size=self.train_batch_size, train_sampler=self.train_sampler,
                          test_data_size=self.test_data_size, test_batch_size=self.test_batch_size,
                          test_sampler=self.test_sampler, data_loader_worker_num=self.data_loader_worker_num)


class MyDataSet(torch.utils.data.Dataset):
    """模型所需要的数据类"""

    def __init__(self, data_set):
        """
        初始化函数
        Args:
            data_set: refer to class DataPrepare.tokenized_data

        """
        self.data_set = data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def train():
    logger = TaskLogger(task_name="ner", multi_gpu=False, log_root=None).root_logger

    # nlp_data moudle
    data_params = configer.DataModuleParams(logger)
    nlp_data_files = DataModule(data_params).load_data()

    # data_helper moudle
    data_helper_params = configer.DataHelperParams(logger)
    data_helper_files = DataHelperModule(data_helper_params, nlp_data_files).load_data()

    # tokenizer moudle
    tokenize_data_params = configer.TokenizerModuleParams(logger)
    tokenize_data_files = TokenizerModule(tokenize_data_params, data_helper_files).load_data()
    with open('/home/user/ll/NLPs/datasets/ner_data/ner_data/test.txt', 'w') as f:

        for x in tokenize_data_files:
            tokens_file = torch.load(x)[0:30]
            for tokens in tokens_file:
                for index, char in enumerate(tokens['input_ids']):
                    if tokens['labels'][index] == 0:
                        if tokens['input_ids'][index] == '。' or tokens['input_ids'][index] == '；' or \
                                tokens['input_ids'][index] == '，':
                            f.writelines(char + ' ' + 'O' + '\n' + '\n')
                        else:
                            f.writelines(char + ' ' + 'O' + '\n')

                    elif tokens['labels'][index] == 1:
                        f.writelines(char + ' ' + 'B-SYM' + '\n')

                    elif tokens['labels'][index] == 2:
                        if index == len(tokens['labels']) - 1:
                            f.writelines(char + ' ' + 'I-SYM' + '\n')

                        elif tokens['labels'][index + 1] == 0:
                            f.writelines(char + ' ' + 'I-SYM' + '\n')

                        elif tokens['labels'][index + 1] == 1:
                            f.writelines(char + ' ' + 'I-SYM' + '\n')

                        elif tokens['labels'][index + 1] == 2:
                            f.writelines(char + ' ' + 'I-SYM' + '\n')

                        else:
                            print("????????")
                            exit()
                    else:
                        print("??")
                        exit()
                f.writelines('\n')

        # for x in tokenize_data_files:
        #     tokens_file = torch.load(x)[120:125]
        #     for tokens in tokens_file:
        #         for index, char in enumerate(tokens['input_ids']):
        #             if tokens['labels'][index] == 0:
        #                 if tokens['input_ids'][index] == '。' or tokens['input_ids'][index] == '；' or \
        #                         tokens['input_ids'][index] == '，':
        #                     f.writelines(char + ' ' + 'O' + '\n' + '\n')
        #                 else:
        #                     f.writelines(char + ' ' + 'O' + '\n')
        #
        #             elif tokens['labels'][index] == 1:
        #                 f.writelines(char + ' ' + 'SYM' + '\n')
        #
        #             elif tokens['labels'][index] == 2:
        #                 if index == len(tokens['labels']) - 1:
        #                     f.writelines(char + ' ' + 'SYM' + '\n')
        #
        #                 elif tokens['labels'][index + 1] == 0:
        #                     f.writelines(char + ' ' + 'SYM' + '\n')
        #
        #                 elif tokens['labels'][index + 1] == 1:
        #                     f.writelines(char + ' ' + 'SYM' + '\n')
        #
        #                 elif tokens['labels'][index + 1] == 2:
        #                     f.writelines(char + ' ' + 'SYM' + '\n')
        #
        #                 else:
        #                     print("????????")
        #                     exit()
        #             else:
        #                 print("??")
        #                 exit()
        #         f.writelines('\n')

    exit()
    # data_loader moudle
    data_loader_params = configer.DataLoaderParams(logger)
    data_loader_obj = DataLoaderModule(data_loader_params, tokenize_data_files).get_data_loader_obj()
    total_train_data_num = data_loader_obj.get_total_train_data_number()


if __name__ == '__main__':
    train()
