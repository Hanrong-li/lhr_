import random
import pandas as pd
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from sklearn.model_selection import train_test_split


class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length=256, memmap=False):
        super().__init__()
        self.max_length = max_length

        if memmap:
            with open(data_path_lst[0], 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize
            self.data = np.memmap(data_path_lst[0], dtype=np.dtype('uint16'),
                                  shape=(flen // max_length, max_length), mode='r')
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length * (len(data) // max_length)]
            self.data = data.reshape(-1, max_length)

        print(f"memmap: {memmap} | train data.shape: {self.data.shape}")
        print("Data loading finished")

        # 设置Jittor Dataset必要属性
        self.total_len = self.data.shape[0]
        self.set_attrs(total_len=self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
        return jt.array(X), jt.array(Y)


# 使用示例
if __name__ == "__main__":
    pass