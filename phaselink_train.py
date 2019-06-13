#! /bin/env python
import numpy as np
import warnings
import os
import torch
import torch.utils.data
import h5py

n_test = 50000
n_epochs = 20
device = torch.device('cuda')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class StackedGRU(torch.nn.Module):
    def __init__(self):
        super(StackedGRU, self).__init__()
        self.hidden_dim = 64
        self.gru1 = torch.nn.GRU(5, self.hidden_dim, \
            batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(self.hidden_dim*2, self.hidden_dim, \
            batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        #self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x):
        out = self.gru1(x)
        h_t = out[0]
        out = self.gru2(h_t)
        h_t = out[0]
        out = self.fc1(h_t)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class Model():
    def __init__(self, network, optimizer, model_path):
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path

    def train(self, train_loader, val_loader, n_epochs):
        from torch.autograd import Variable
        import time

        loss = torch.nn.BCELoss()
        n_batches = len(train_loader)
        training_start_time = time.time()

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_acc = 0
            running_val_acc = 0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            total_val_loss = 0
            total_val_acc = 0
            running_sample_count = 0

            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, labels = data
                inputs, labels = Variable(
                    inputs.cuda(device)), Variable(
                    labels.cuda(device))

                # Set gradients for all parameters to zero
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs)

                # Backward pass
                outputs = outputs.view(-1)
                labels = labels.view(-1)
                loss_value = loss(outputs, labels.float())
                loss_value.backward()

                # Update parameters
                self.optimizer.step()

                # Print statistics
                running_loss += loss_value.data.item()
                total_train_loss += loss_value.data.item()

                # Calculate categorical accuracy
                pred = torch.round(outputs).long()

                running_acc += (pred == labels).sum().item()
                running_sample_count += len(labels)

                # Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                        "train_acc: {:4.2f}% took: {:.2f}s".format(
                        epoch + 1, int(100 * (i + 1) / n_batches),
                        running_loss / print_every,
                        100*running_acc / running_sample_count,
                        time.time() - start_time))

                    # Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            running_sample_count = 0
            for inputs, labels in val_loader:

                # Wrap tensors in Variables
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)
                #inputs, labels = Variable(
                #    inputs.cuda()), Variable(
                #    labels.cuda())

                # Forward pass only
                val_outputs = self.network(inputs)
                val_outputs = val_outputs.view(-1)
                labels = labels.view(-1)
                val_loss = loss(val_outputs, labels.float())
                total_val_loss += val_loss.data.item()

                # Calculate categorical accuracy
                pred = torch.round(val_outputs).long()
                running_val_acc += (pred == labels).sum().item()
                running_sample_count += len(labels)

            total_val_loss /= len(val_loader)
            total_val_acc = running_val_acc / running_sample_count
            print(
                "Validation loss = {:.4e}   acc = {:4.2f}%".format(
                    total_val_loss,
                    100*total_val_acc))

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': total_val_loss,
            }, '%s/model_%03d_%f.pt' % (self.model_path, epoch, total_val_loss))

        print(
            "Training finished, took {:.2f}s".format(
                time.time() -
                training_start_time))

    def predict(self, data_loader):
        from torch.autograd import Variable
        import time

        for inputs, labels in val_loader:

            # Wrap tensors in Variables
            inputs, labels = Variable(
                inputs.cuda(device)), Variable(
                labels.cuda(device))

            # Forward pass only
            val_outputs = self.network(inputs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("phaselink_train config_json")
        sys.exit()

    with open(sys.argv[1], "r") as f:
        params = json.load(f)

    with h5py.File(params['hdf_file'], 'r') as f:
        X = f['X'][:1200000]
        Y = f['Y'][:1200000]
    print(X.shape, Y.shape)

    dataset = MyDataset(X, Y)

    n_samples = len(dataset)
    indices = list(range(n_samples))

    validation_idx = np.random.choice(indices, size=n_test, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        sampler=validation_sampler
    )

    stackedgru = StackedGRU()
    #stackedgru = torch.nn.DataParallel(StackedGRU())
    stackedgru = stackedgru.cuda(device)
    optimizer = torch.optim.Adam(stackedgru.parameters())
    #optimizer = torch.optim.SGD(ConvNet.parameters(), lr=0.01)
    model = Model(stackedgru, optimizer, \
        model_path='/export/data1/zross/phaselink/models_saved')
    model.train(train_loader, val_loader, n_epochs)
