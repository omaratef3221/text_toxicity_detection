from get_data import get_data
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
obj = get_data()
# Print data details
print(obj.get_len())
print(obj.get_summary())
# Get data loaders
train_dl, test_dl = obj.prepare_data()


