import pandas as pd
from pathlib import Path
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.utils import resample


DATAPOINTS_IN_EPOCH = 3000  # 30 sec * 100Hz

class2idx = {
    'R': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    'W': 4,
}

def preprocess_data(df):
    '''
        stages are one of: W, R, 1, 2, 3, 4, M (Movement time) and ? (not scored)
    '''
    #df = df.loc[df['subject'].isin(['SC4001', 'SC4002', 'SC4011', 'SC4012','SC4021','SC4022','SC4031','SC4032','SC4041','SC4042',
    #                               'SC4051', 'SC4052', 'SC4061', 'SC4062','SC4071','SC4072','SC4081','SC4082','SC4091','SC4092'])]
    print(df.subject.unique())
    print("Starting with {} epochs".format(len(df)))
    # combine N3 and N4 into N3 stage
    df.loc[df['annotation'] == '4', 'annotation'] = '3'

    # exclude 'M' and '?' epochs
    df = df.loc[df['annotation'] != '?']
    df = df.loc[df['annotation'] != 'M']
    
    # remove epochs that don't have all 3000 data points
    df = df.loc[df['EEG_Fpz-Cz_uV'].map(len) == DATAPOINTS_IN_EPOCH]
    print("After removing non 30 second: {} epochs".format(len(df)))
    
    # TODO remove 'W' stages that are more than 30 minutes from any sleep stage
    
    # resample based on label frequency
    print("before resampling: ", df.groupby('annotation').size())
    g = df.groupby(by='annotation', group_keys=False)
    df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True))

    #df1 = df[df['annotation'] == 'W'].sample(frac=.1)
    #df = df.drop(df1.index)
    print("after resampling: ", df.groupby('annotation').size())
    
    # encode class label
    df['annotation'] = df['annotation'].replace(class2idx)
    print("Finsihed processing data, {} subjects, {} epochs in total".format(len(df.subject.unique()), len(df)))
    return df.reset_index(drop=True)


def get_label_stats(df):
    return df.groupby(['annotation']).agg(['count'])


def build_dataset(df):
    #X = torch.FloatTensor(df['EEG_Fpz-Cz_uV']) # (num_epoch, 3000)
    X = torch.FloatTensor(channels_np) # (64165, 3, 3000)
    print(X.shape)
    
    # X = torch.unsqueeze(X, 1) # add one more dimesion for channel -> (num_epoch, 1, 3000)
    Y = torch.tensor(df['annotation'].values)

    dataset = TensorDataset(X, Y)
    train_sample = int(len(df) * 0.70)
    val_sample = len(df) - train_sample
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_sample, val_sample])
    print("Size of train_dataset:", len(train_dataset))
    print("Size of val_dataset:", len(val_dataset))
    
    return train_dataset, val_dataset

def load_data(train_dataset, val_dataset):
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print("# of train batches: {}, # of val batches: {}".format(len(train_loader), len(val_loader)))
    return train_loader, val_loader

class ConvModelA(nn.Module):

    def __init__(self, sampling_rate):
        super(ConvModelA, self).__init__()
        
        self.fs = sampling_rate
        self.conv1 = nn.Conv1d(in_channels = 3,
                                out_channels = 16,    # specified in paper
                                kernel_size= int(self.fs / 2),  # specified in paper
                                stride = int(self.fs / 16))     # specified in paper
        self.batch_norm = nn.BatchNorm1d(num_features = 16)
        self.maxpool = nn.MaxPool1d(kernel_size = 8, stride=8)
        self.dropout = nn.Dropout(p=0.5)
        
        self.conv2 = nn.Conv1d(in_channels = 16,       
                                out_channels = 128,    
                                kernel_size= 8,
                                stride = int(self.fs / 16))
        self.batch_norm2 = nn.BatchNorm1d(num_features = 128)

        self.maxpool2 = nn.MaxPool1d(kernel_size = 4, stride=4)
        self.fc = nn.Linear(in_features= 128*2, 
                            out_features = 5) # 5 categories
        
    def forward(self,x):
        # input is of shape (batch_size=32, num_channel=3, DATAPOINTS_IN_EPOCH=3000)
        x = self.conv1(x)   # L_out = (3000 - 49 - 1) / 6 + 1 = 492
                            # output: (batch_size=32, num_channel=16, L_out=492)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.maxpool(x) # input size: (batch_size=32, num_channel=16, L_out=492)
                            # output size: (batch_size=32, num_channel=16, L_out=(492-(8-1)-1)/8+1=61)
        x = self.dropout(x) # [32, 16, 61] no shape change
        #conv2
        x = self.conv2(x)   # L_out = 
                            # output: (batch_size=32, num_channel=128, L_out=9)
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        #maxpool
        x = self.maxpool2(x) # output size: (batch_size=32, num_channel=128, L_out=2)
        # linear
        x = x.view(-1, 128*2)
        x = self.fc(x)      # output size: (batch_size=32, class=5)
        return x
        
class ConvModelB(nn.Module):

    def __init__(self):
        super(ConvModelB, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=8)
        self.batch_norm1 = nn.BatchNorm1d(num_features = 8)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8)
        self.batch_norm2 = nn.BatchNorm1d(num_features = 16)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8)
        self.batch_norm3 = nn.BatchNorm1d(num_features = 32)
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=8)
        
        self.fc = nn.Linear(in_features= 128, out_features = 5)
  
    def forward(self, x):
        # input is of shape (batch_size=32, num_channel=3, DATAPOINTS_IN_EPOCH=3000)
       
        # CNN needs input of [N = num_epochs, Cin = 3, Lin = 3000 (1D signal with 3000 reading) ]      

        x = self.conv1(x) 
        x = self.batch_norm1(x)
        x = F.relu(x) # [32, 8, 2993]
        x = self.maxpool(x)  # [batch_size, 8, 374]

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x) # [32, 16, 367]
        x = self.maxpool(x)  # [batch_size, 16, 45]

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x) # [32, 32, 38]
        x = self.maxpool(x)  # [batch_size, 32, 4]

        x = torch.reshape(x, (-1, 32*4))  # flatten: # [batch_size, 128]
        x = self.fc(x) 
        return x  # [batch_size, 5]

def train_model(model, train_dataloader, n_epoch, optimizer, criterion):
    model.train()
    for epoch in range(n_epoch):
        curr_loss = []
        for x, y in train_dataloader:
            out = model(x)
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
    for x, y in val_dataloader:
        out = model(x)
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
    # auroc = roc_auc_score(y_true, y_pred)
    average_f1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    kappa = cohen_kappa_score(y_true, y_pred)
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred))
    print(model)
    print("y_pred", y_pred[:10])
    print("y_true", y_true[:10])
    print(f"accuracy_score: {acc}")
    print(f"average_f1: {average_f1}")
    print(f"cohen_kappa_score: {kappa}")
    print(f"confusion_matrix:\n {confusion}")

    percision_recall_df = pd.DataFrame(per_class_percision.reshape(1,5), index = ['percision'], columns =['R', 'N1', 'N2', 'N3', 'W'])
    percision_recall_df.loc['recall'] = per_class_recall
    percision_recall_df.loc['F1'] = per_class_f1
    return percision_recall_df, confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
	three_channels = pd.read_parquet('data/stat/output_full_partitioned', columns=['annotation', 'EEG_Fpz-Cz_uV','EEG_Pz-Oz_uV','EOG_horizontal_uV'])
	processed_df = preprocess_data(three_channels)
	channels = processed_df[['EEG_Fpz-Cz_uV', 'EEG_Pz-Oz_uV', 'EOG_horizontal_uV']].to_numpy()
	channels_np = np.array(channels.tolist())
	train_dataset, val_dataset = build_dataset(processed_df)
	train_loader, val_loader = load_data(train_dataset, val_dataset)
	conv_model_a = ConvModelA(100)   # sampling_rate = 100Hz
	# conv_model_b = ConvModelB()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(conv_model_a.parameters(), lr=0.001)
	conv_model_a = train_model(conv_model_a, train_loader, 20, optimizer, criterion)

	percision_recall_df, confusion_matrix1 = get_evaluation_result(normal_model, val_loader)
	print(percision_recall_df)








