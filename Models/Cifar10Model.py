import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.IModel import IModel


class Cifar10NNModel(IModel):
    def __init__(self):
        super(Cifar10NNModel, self).__init__()

        self.class_names = ["Airplane",
                            "Automobile",
                            "Bird",
                            "Cat",
                            "Deer",
                            "Dog",
                            "Frog",
                            "Horse",
                            "Ship",
                            "Truck"]

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    def get_class_name(self, class_id) -> str:
        return self.class_names[class_id]
