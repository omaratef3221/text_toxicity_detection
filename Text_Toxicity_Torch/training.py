from torch.optim import Adam
from tqdm import tqdm
from torch.nn import MSELoss
import torch

def train(model, data_object, learning_rate, epochs):
    # Get data loaders
    train_dl, test_dl = data_object.prepare_data()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") #use CPU if GPU is not available
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    for epochs in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dl):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dl): .3f} \
            | Train Accuracy: {total_acc_train / len(train_dl): .3f}')