import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer
class get_data:
    def __init__(self):
        self.data = pd.read_csv("train_data_version3.csv")
        self.x = self.data["text"]
        self.y = self.data["y"]
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
        X_train, X_test, y_train, y_test = train_test_split(self.data["text"], self.data["y"], test_size=0.25)
        X_train = [self.tokenizer(text,padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in X_train]
        X_test = [self.tokenizer(text, padding='max_length', max_length=512, truncation=True,
                                  return_tensors="pt") for text in X_test]
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        train_dl = DataLoader([X_train, y_train], batch_size=16, shuffle=True)
        test_dl = DataLoader([X_test, y_test], batch_size=1024, shuffle=False)
        return train_dl, test_dl