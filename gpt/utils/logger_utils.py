# -*-coding:utf-8-*-

import logging
import time


class TaskLogger:
    def __init__(self, task_name, multi_gpu=False, log_root=None):
        self.task_name = task_name
        self.log_root = log_root

        import torch.distributed

        if multi_gpu:
            torch.distributed.init_process_group(backend="nccl")
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0
        self.root_logger = self.__init_root_logger()

    def __init_root_logger(self):
        if self.rank == 0:
            logger = logging.getLogger()
        else:
            logger = logging.getLogger("child")

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"%(asctime)s - {self.task_name} - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

        handler.setFormatter(formatter)
        if logger.handlers:
            [logger.removeHandler(hd) for hd in logger.handlers]
        logger.addHandler(handler)
        if self.rank == 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARN)

        return logger


if __name__ == '__main__':
    pass
