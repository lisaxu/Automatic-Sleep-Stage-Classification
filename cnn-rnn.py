import pandas as pd
from pathlib import Path
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.utils import resample
from icecream import ic
from torch.utils.data import Dataset

DATAPOINTS_IN_EPOCH = 3000  # 30 sec * 100Hz
class2idx = {
    'R': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    'W': 4,
    }

class CustomEEGDataset(Dataset):
    
    def __init__(self, data_dir, pad=True):
        self.data_dir = data_dir
        self.pad = pad
        #self.epochs = None
        #self.labels = None
        self.num_sleep = 150 # number of sleeps here (each patient has two sleeps)
        self.max_epochs = 2795 # TODO get this stats before building the dataset
    
    def __len__(self):
        return self.num_sleep
    
    def preprocess_data(self, df):
        '''
            stages are one of: W, R, 1, 2, 3, 4, M (Movement time) and ? (not scored)
        '''
        # combine N3 and N4 into N3 stage
        df.loc[df['annotation'] == '4', 'annotation'] = '3'

        # exclude 'M' and '?' epochs
        df = df.loc[df['annotation'] != '?']
        df = df.loc[df['annotation'] != 'M']

        # remove epochs that don't have all 3000 data points
        df = df.loc[df['EEG_Fpz-Cz_uV'].map(len) == DATAPOINTS_IN_EPOCH]

        # encode class label
        df['annotation'] = df['annotation'].replace(class2idx)
        return df

    def __getitem__(self, index):
        # read parquet file
        parquet_file = f"{self.data_dir}/{index}"
        df = pd.read_parquet(parquet_file, columns=['annotation', 'EEG_Fpz-Cz_uV','EEG_Pz-Oz_uV','EOG_horizontal_uV'])
        processed_df = self.preprocess_data(df)

        # channels is of shape (# epoch, 3(channel), 3000)
        # label is of shape (# epoch, 1)
        channels = processed_df[['EEG_Fpz-Cz_uV', 'EEG_Pz-Oz_uV', 'EOG_horizontal_uV']].to_numpy()
        channels_np = np.array(channels.tolist())  # use to get the array object out of the way
        labels_np = processed_df[['annotation']].to_numpy()
        
        # create padding
        epoch_len = channels_np.shape[0]
        padding_epoch = self.max_epochs - epoch_len

        if padding_epoch and self.pad:
            padding = np.zeros((padding_epoch,3,3000))
            channels_np = np.concatenate((channels_np, padding), axis=0)
            labels_np = np.concatenate((labels_np, np.zeros((padding_epoch,1))), axis=0)
            
        return {'epoch_len': [epoch_len],
                'X': torch.tensor(channels_np, dtype=torch.float),  # [max_epochs, 3, 3000]
                'Y': torch.tensor(labels_np, dtype=torch.long)}      # [max_epochs, 1]
        
def load_data(dataset, batch_size=16):
    
    # split dataset into train and test
    split = int(len(dataset)*0.7)
    lengths = [split, len(dataset) - split]
    train_dataset, val_dataset = random_split(dataset, lengths)
    print("Total # sleeps: {}, train: {}, validate: {}".format(len(dataset), len(train_dataset), len(val_dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0) # num_workers is for multi-process data loading
    
    print("train batches: {}, val batches: {}".format(len(train_loader), len(val_loader)))
    return train_loader, val_loader

class ConvPlusRecurrent(nn.Module):

    def __init__(self, max_epochs):
        super(ConvPlusRecurrent, self).__init__()

        self.max_epochs = max_epochs

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=8)    
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8)    
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8)    
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=8)
        
        self.lstm = nn.LSTM(input_size = 128,hidden_size = 32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(in_features= 32, out_features = 5)
 
    def forward(self, x, epoch_len):
        # input is of shape (batch_size, max_epochs=2795, num_channel=3, DATAPOINTS_IN_EPOCH=3000)

        # reshape to [batch_size*max_epochs, 3, 3000]
        # so that each epoch can through CNN
        # CNN needs input of [N = num_epochs, Cin = 3, Lin = 3000 (1D signal with 3000 reading) ]      
        length = x.shape[0]
        x = x.view((length*self.max_epochs, 3, 3000))  # [num_epochs, 3, 3000]

        x = self.conv1(x)   
        x = F.relu(x)
        x = self.maxpool(x)  # [num_epochs, 8, 374]

        x = self.conv2(x)   
        x = F.relu(x)
        x = self.maxpool(x)  # [num_epochs, 16, 45]
        
        x = self.conv3(x)   
        x = F.relu(x)
        x = self.maxpool(x)  # [num_epochs, 32, 4]
        
        x = torch.reshape(x, (-1, 32*4))  # flatten: [num_epochs, 32*4=128]
        x = torch.reshape(x, (length, self.max_epochs, -1))  # reshape into [batch, max_epochs, 128]
        
        # LSTM input should be of shape (batch, seq(max_epochs), feature)
        output, hidden = self.lstm(x) # [16, 2795, 64]        
        x = self.fc(output) 
        
        return x  # [16, 2795, 5]

def get_real_epochs(x, y, epoch_len):
    # apply mask (epoch_len) and gather the labels from only the true epochs
    # input: x [batch_size, 2795, 5]
    # input: y [batch_size, 2795, 1]
    # input: epoch_len [1, batch_size]
    epoch_len = epoch_len[0]
    predicted_labels = []
    true_labels = []

    for i in range(x.shape[0]):
        real_num_epoch = int(epoch_len[i])
        predicted_labels.append(x[i][:real_num_epoch])  # [n, 5]
        true_labels.append(y[i][:real_num_epoch])  # [n, 1]
        
    predicted_labels = torch.cat(predicted_labels) #[N, 5]
    true_labels = torch.cat(true_labels).squeeze() #[N, 1] -> [N]

    return predicted_labels, true_labels

def train_model(model, train_dataloader, n_epoch, optimizer, criterion):
    model.train()
    for epoch in range(n_epoch):
        curr_loss = []
        for values in train_dataloader:
            x = values['X']
            y = values['Y'] 
            epoch_len = values['epoch_len']

            out = model(x, epoch_len)
            out, y = get_real_epochs(out, y, epoch_len)
            
            # corss entropy loss
            # "input is expected to contain raw, unnormalized scores for each class"
            # out: [N, class=5]
            # y:   [N]
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_loss.append(loss.cpu().data.numpy())
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_loss)}")
    return model

def eval_model(model, val_dataloader):
    model.eval()
    Y_pred = []
    Y_true = []
    for values in val_dataloader:
        x = values['X']
        y = values['Y'] 
        epoch_len = values['epoch_len']
        
        out = model(x, epoch_len)
        out, y = get_real_epochs(out, y, epoch_len)

        _, pred = torch.max(input=out, dim=1)
        Y_pred.append(pred.detach().numpy())
        Y_true.append(y.detach().numpy())

    Y_pred = np.concatenate(Y_pred, axis=0) # stack the rows from different batches
    Y_true = np.concatenate(Y_true, axis=0)
    return Y_pred, Y_true

def get_evaluation_result(model, val_loader):
    y_pred, y_true = eval_model(model, val_loader)
    acc = accuracy_score(y_true, y_pred)
    per_class_percision = precision_score(y_true, y_pred, average=None) # average='None' to return per-class percision
    per_class_recall = recall_score(y_true, y_pred, average=None)

    average_f1 = f1_score(y_true, y_pred, average='weighted')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    kappa = cohen_kappa_score(y_true, y_pred)
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred))

    print("y_pred", y_pred)
    print("y_true", y_true)
    print(f"accuracy_score: {acc}")
    print(f"average_f1: {average_f1}")
    print(f"cohen_kappa_score: {kappa}")
    print(f"confusion_matrix:\n {confusion}")

    percision_recall_df = pd.DataFrame(per_class_percision.reshape(1,5), index = ['percision'], columns =['R', 'N1', 'N2', 'N3', 'W'])
    percision_recall_df.loc['recall'] = per_class_recall
    percision_recall_df.loc['F1'] = per_class_f1
    return confusion_matrix(y_true, y_pred,normalize = 'true'), percision_recall_df

if __name__ == "__main__":
	custom_eeg_dataset = CustomEEGDataset('data/output_full_partitioned')
	train_loader, val_loader = load_data(custom_eeg_dataset, 6)
	model = ConvPlusRecurrent(max_epochs = 2795)
	criterion = nn.CrossEntropyLoss()#weight=torch.FloatTensor([1,5,1,1,1])
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	model = train_model(model, train_loader, 30, optimizer, criterion)
	confusion, percision_recall_df = get_evaluation_result(model, val_loader)
	print(percision_recall_df)
            
