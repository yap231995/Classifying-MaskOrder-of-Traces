import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(5000, 500, 3, padding=1)
        self.conv2 = nn.Conv1d(500, 100, 3, padding=1)
        self.conv3 = nn.Conv1d(100, 50, 3, padding=1)
        self.fc1 = nn.Linear(1000 , 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
