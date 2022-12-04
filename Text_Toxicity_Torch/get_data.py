import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
class get_data(Dataset):
    def __init__(self):
        self.data = pd.read_csv("train_data_version3.csv")
        self.x = list(self.data["text"])
        self.y = list(self.data["y"])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_df(self):
        return self.data
    def get_len(self):
        return self.data.shape
    def get_item(self, indx):
        return self.data.loc[indx]
    def get_summary(self):
        return self.data.describe()
    def prepare_data(self):
        #X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25)
        self.texts = [self.tokenizer(text,
                                padding='max_length', max_length= 512, truncation=True,
                                return_tensors="pt") for text in self.x]

        train, test = torch.utils.data.random_split([self.texts, self.y], [0.7, 0.3])



        # y_train = torch.tensor(list(y_train), dtype=torch.float32)
        # y_test = torch.tensor(list(y_test), dtype=torch.float32)
        train_dl = DataLoader(train, batch_size=16, shuffle=True)
        test_dl = DataLoader(test, batch_size=16, shuffle=False)
        return train_dl, test_dl