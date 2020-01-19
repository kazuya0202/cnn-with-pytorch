import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cnn

# import datasets as ds
import torch_datasets as td
from global_variables import GlobalVariables


if __name__ == "__main__":
    gv = GlobalVariables()
    params = [
        gv.image_path,  # path
        gv.extensions,  # extensions
        gv.test_size,  # test_size
    ]
    ds = td.CreateDataset(*params)

    size = 80
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])

    # create train dataset
    train_dataset = td.CustomDataset(ds, 'train', transform)
    train_data = DataLoader(train_dataset, batch_size=4, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = cnn.Net()
    net.to(device)
    net.zero_grad()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    # === TRAIN ====
    def train(epoch_num):
        for ep in range(epoch_num):
            loss_sum = 0
            print(f'\ntrain epoch: {ep}')

            net.train()
            for idx, (data, label) in enumerate(train_data):
                optimizer.zero_grad()
                out = net(data.to(device))
                loss = criterion(out.to(device), label.to(device))
                loss.backward()
                optimizer.step()

                loss_sum += loss

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch_num, idx * len(data), len(train_data.dataset),
                #     100. * idx / len(train_data), loss.data[0]))

                # print(f'\rtrain epoch: {idx} - loss: {loss.data}')
                print(f'\rloss: {loss.data}', end='')
            print(f'\nloss_sum: {loss_sum / len(train_data.dataset)}')

    train(5)

    # create test dataset
    test_dataset = td.CustomDataset(ds, 'test', transform)
    test_data = DataLoader(test_dataset, shuffle=True)

    # === TEST ===
    def test():
        net.eval()
        # test_loss = 0
        # correct = 0

        for data, label in test_data:
            out = net(data.to(device))
            # test_loss += criterion(out.to(device), label.to(device)).data
            pred = out.data.max(1, keepdim=True)[1]
            print(pred[0], '/ label:', label)
            # correct += pred.eq(label.data.view_as(pred)).long().to(device).sum()

        # test_loss /= len(test_data.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_data.dataset),
        #     100. * correct / len(test_data.dataset)))

    print('\n\ntest ---')
    test()
