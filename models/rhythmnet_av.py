import torch
import torch.nn as nn
import torchvision.models as models


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        in_features = 1000
        # DO-NOT --- remove the FC layer
        # self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.rnn = nn.GRU(
            input_size=in_features,
            hidden_size=in_features,
        )

        self.fc_wo_RNN = nn.Linear(in_features, 1)
        self.fc_RNN = nn.Linear(in_features, 1)

    def forward(self, x):
        # can only use batch size-1
        num_frames, c, h, w = x.shape
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)

        # w/o - RNN
        x_wo_rnn = self.fc_wo_RNN(x)

        # w/ RNN
        x = x.view(num_frames, -1)
        x_hn, _ = self.rnn(x)
        x_hn = self.fc_RNN(x_hn)
        return x_wo_rnn.squeeze(1), x_hn.squeeze(1)
