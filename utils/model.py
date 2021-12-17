import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


class INIT_CLIP_LSTM(nn.Module):

    def __init__(self):
        super(INIT_CLIP_LSTM, self).__init__()

        self.LSTM = nn.LSTM(input_size = 512,
                            hidden_size = 64,
                            num_layers = 2,
                            dropout=0.3,
                            batch_first = True,
                            bidirectional = True)

        self.fc1 = nn.Linear(in_features = 64,
                             out_features = 32)

        self.fc2 = nn.Linear(in_features = 32,
                            out_features = 14)


    def forward(self, feat):


        out, (h_0, c_0) = self.LSTM(feat)

        out = self.fc1(h_0.mean(0))

        out = F.softmax(self.fc2(out), dim=1)

        return out




class CLIP_LSTM(nn.Module):

    def __init__(self):
        super(CLIP_LSTM, self).__init__()

        self.LSTM = nn.LSTM(input_size = 512,
                            hidden_size = 64,
                            num_layers = 2,
                            dropout=0.3,
                            batch_first = True,
                            bidirectional = True)

        self.bn = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(in_features = 64,
                             out_features = 32)



        self.fc2 = nn.Linear(in_features = 32,
                            out_features = 14)


    def forward(self, feat):

        self.LSTM.flatten_parameters()

        out, (h_0, c_0) = self.LSTM(feat)

        out = F.dropout(self.fc1(self.bn(h_0.mean(0))), p=0.3)


        out = F.softmax(self.fc2(out), dim=1)


        return out


class ANOMALY_CLIP_LSTM(nn.Module):

    def __init__(self):
        super(ANOMALY_CLIP_LSTM, self).__init__()

        self.LSTM = nn.LSTM(input_size = 512,
                            hidden_size = 64,
                            num_layers = 2,
                            dropout=0.3,
                            batch_first = True,
                            bidirectional = True)

        self.fc1 = nn.Linear(in_features = 64,
                             out_features = 32)

        self.fc2 = nn.Linear(in_features = 32,
                            out_features = 2)


    def forward(self, feat):


        out, (h_0, c_0) = self.LSTM(feat)

        out = F.dropout(self.fc1(h_0.mean(0)), 0.2)

        out = F.softmax(self.fc2(out), dim=1)

        return out


class ANOMALY_CLIP_LSTM2(nn.Module):

    def __init__(self):
        super(ANOMALY_CLIP_LSTM2, self).__init__()

        self.LSTM = nn.LSTM(input_size = 512,
                            hidden_size = 64,
                            num_layers = 2,
                            dropout=0.3,
                            batch_first = True,
                            bidirectional = True)

        self.bn = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(in_features = 64,
                             out_features = 32)

        self.fc2 = nn.Linear(in_features = 32,
                            out_features = 16)

        self.fc3 = nn.Linear(in_features = 16,
                            out_features = 2)



    def forward(self, feat):


        out, (h_0, c_0) = self.LSTM(feat)

        out = F.dropout(self.fc1(self.bn(h_0.mean(0))), 0.3)

        out = F.dropout(self.fc2(out), 0.3)

        out = F.softmax(self.fc3(out), dim=1)

        return out


class ANOMALY_CLIP_LSTM3(nn.Module):

    def __init__(self):
        super(ANOMALY_CLIP_LSTM3, self).__init__()

        self.LSTM = nn.LSTM(input_size = 512,
                            hidden_size = 64,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = True)

        self.do = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_features = 64,
                             out_features = 2)


    def forward(self, feat):


        out, (h_0, c_0) = self.LSTM(feat)

        out = F.softmax(self.fc1(self.do(h_0.mean(0))), dim=1)


        return out
