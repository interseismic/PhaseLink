#! /bin/env python
import numpy as np
import os
import torch
import torch.utils.data
import sys
import json
import pickle

n_epochs = 20

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
        self.hidden_size = 128
        self.fc1 = torch.nn.Linear(5, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(2*128, 1)
        self.gru1 = torch.nn.GRU(32, self.hidden_size, \
            batch_first=True, bidirectional=True)
        self.gru2 = torch.nn.GRU(self.hidden_size*2, self.hidden_size, \
            batch_first=True, bidirectional=True)
        self.sigmoid = torch.nn.Sigmoid()
        #self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, inp):
        out = self.fc1(inp)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        out = torch.nn.functional.relu(out)
        out = self.gru1(out)
        h_t = out[0]
        out = self.gru2(h_t)
        h_t = out[0]
        out = self.fc4(h_t)
        out = self.sigmoid(out)
        return out

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, self.hidden_size)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden

class AttnDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
            max_length):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        # Layers
        self.attn = torch.nn.Linear(self.hidden_size + 1, self.max_length)
        self.attn_combine = torch.nn.Linear(self.hidden_size + 1,
            self.hidden_size)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs):
        import torch.nn.functional as F
        cat_inp = torch.cat((input, hidden), 2)
        attn_weights = F.softmax(self.attn(cat_inp), dim=2)
        attn_applied = torch.bmm(attn_weights.permute(1,0,2),
            encoder_outputs.permute(1,0,2))

        output = torch.cat((input.permute(1,0,2), attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output).permute(1,0,2)

        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = torch.sigmoid(output)
        return output, hidden

class AttnModel():
    def __init__(self, encoder, decoder, enc_optimizer, dec_optimizer,
        model_path):
        self.encoder = encoder
        self.decoder = decoder
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self.model_path = model_path

    def train(self, train_loader, val_loader, n_epochs):
        from torch.autograd import Variable
        import time

        loss = torch.nn.BCELoss()
        n_batches = len(train_loader)
        training_start_time = time.time()
        SOS_token = 0

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
            loss_value = 0

            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, labels = data
                inputs = inputs.to(device).permute(1, 0, 2)
                labels = labels.to(device)

                # Set gradients for all parameters to zero
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()

                # Forward pass
                encoder_outputs, encoder_hidden = self.encoder(inputs)

                decoder_hidden = encoder_hidden
                decoder_input = torch.zeros((1, inputs.shape[1], 1),
                    device=device)

                outputs = torch.zeros(labels.shape, device=device)
                for j in range(labels.shape[0]):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)

                    loss_value += loss(decoder_output, labels[:,j].float())
                    decoder_input = labels[:,j][None,:,None].float().detach()
                    decoder_output = decoder_output.detach()
                    outputs[:,j] = decoder_output.squeeze().detach()

                # Backward pass
                outputs = outputs.view(-1)
                labels = labels.view(-1)
                loss_value.backward(retain_graph=True)


                # Update parameters
                self.enc_optimizer.step()
                self.dec_optimizer.step()

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
            y_pred_all, y_true_all = [], []
            for inputs, labels in val_loader:
                # Wrap tensors in Variables
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)

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

                y_pred_all.append(pred.cpu().numpy().flatten())
                y_true_all.append(labels.cpu().numpy().flatten())

            y_pred_all = np.concatenate(y_pred_all)
            y_true_all = np.concatenate(y_true_all)

            from sklearn.metrics import classification_report
            print(classification_report(y_true_all, y_pred_all))

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
                inputs = inputs.to(device)
                labels = labels.to(device)

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
            y_pred_all, y_true_all = [], []
            for inputs, labels in val_loader:
                # Wrap tensors in Variables
                inputs = inputs.cuda(device)
                labels = labels.cuda(device)

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

                y_pred_all.append(pred.cpu().numpy().flatten())
                y_true_all.append(labels.cpu().numpy().flatten())

            y_pred_all = np.concatenate(y_pred_all)
            y_true_all = np.concatenate(y_true_all)

            from sklearn.metrics import classification_report
            print(classification_report(y_true_all, y_pred_all))

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

    device = torch.device(params["device"])

    X = np.load(params["training_dset_X"])
    Y = np.load(params["training_dset_Y"])
    print(X.shape, Y.shape)

    print(np.where(Y==1)[0].size, "1 labels")
    print(np.where(Y==0)[0].size, "0 labels")

    dataset = MyDataset(X, Y)

    n_samples = len(dataset)
    indices = list(range(n_samples))

    n_test = int(0.1*X.shape[0])

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
        batch_size=32,
        shuffle=False,
        sampler=validation_sampler
    )

    stackedgru = StackedGRU()
    #stackedgru = torch.nn.DataParallel(StackedGRU(), device_ids=['cuda:2', 'cuda:3', 'cuda:4'])
    stackedgru = stackedgru.cuda(device)
    optimizer = torch.optim.Adam(stackedgru.parameters())
    #enc = Encoder(input_size=4, hidden_size=64).cuda(device)
    #dec = AttnDecoder(input_size=4, hidden_size=64, output_size=1,
    #    max_length=500).cuda(device)
    #enc_opt = torch.optim.Adam(enc.parameters())
    #dec_opt = torch.optim.Adam(dec.parameters())

    model = Model(stackedgru, optimizer, \
        model_path='./models_saved/stacked_gru/')
    #model = AttnModel(enc, dec, enc_opt, dec_opt, \
    model.train(train_loader, val_loader, n_epochs)
