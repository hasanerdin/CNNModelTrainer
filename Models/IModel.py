import torch.nn as nn
from abc import abstractmethod


class IModel(nn.Module):
    def __init__(self):
        super(IModel, self).__init__()
        self.class_names = None

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_class_name(self, class_id) -> str:
        pass
