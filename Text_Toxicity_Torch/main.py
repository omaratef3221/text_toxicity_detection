from get_data import get_data
from model import BertClassifier
from training import train

obj = get_data()
# Print data details
print(obj.get_len())
print(obj.get_summary())
print(obj.prepare_data())
# Main
EPOCHS = 25
model = BertClassifier()
LR = 1e-4
train(model, obj, learning_rate= LR, epochs= EPOCHS)



