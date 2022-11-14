import torch.nn as nn
import torch.nn.functional as F

from Models.IModel import IModel


class MNISTModel(IModel):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.class_names = range(10)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        self.max_pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    def get_class_name(self, class_id: int) -> str:
        return str(class_id)
