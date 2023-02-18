from data_helper.get_data import PretrainData
from utils.logger_utils import TaskLogger
from data_helper.data_cleaning import DataCleaning
from data_helper.tokenizer import DataTokenizer
import torch
from tqdm import tqdm


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
    logger = TaskLogger(task_name="pretrain", multi_gpu=False, log_root=None).root_logger
    _test = DataTokenizer(logger).insert_case_data()
    exit()
    # # nlp_data moudle
    # data_params = configer.DataModuleParams(logger)
    # nlp_data_files = DataModule(data_params).load_data()
    #
    # # data_helper moudle
    # data_helper_params = configer.DataHelperParams(logger)
    # data_helper_files = DataHelperModule(data_helper_params, nlp_data_files).load_data()
    #
    # # tokenizer moudle
    # tokenize_data_params = configer.TokenizerModuleParams(logger)
    # tokenize_data_files = TokenizerModule(tokenize_data_params, data_helper_files).load_data()

    # # data_loader moudle
    # data_loader_params = configer.DataLoaderParams(logger)
    # data_loader_obj = DataLoaderModule(data_loader_params, tokenize_data_files).get_data_loader_obj()
    # total_train_data_num = data_loader_obj.get_total_train_data_number()


if __name__ == '__main__':
    train()
