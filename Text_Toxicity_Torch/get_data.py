import pandas as pd

class get_data:
    def __init__(self):
        self.data = pd.read_csv("train_data_version3.csv")
        self.x = self.data["text"]
        self.y = self.data["y"]
    def get_len(self):
        return self.data.shape
    def get_item(self, indx):
        return self.data.loc[indx]
    def get_summary(self):
        return self.data.describe()