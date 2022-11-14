from enum import Enum
import torch

from Models.IModel import IModel


class OptimizerNames(Enum):
    SGD = 1
    SGDWithMomentum = 2
    RMSProp = 3
    Adam = 4

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class CriterionNames(Enum):
    BCELoss = 1
    CrossEntropyLoss = 2

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class TrainerConfig:
    epochs = 10
    batch_size = 32
    lr = 3e-4

    resume = False
    optimizer_name = OptimizerNames.SGD
    criterion_name = CriterionNames.CrossEntropyLoss

    momentum = 0.9
    betas = (0.9, 0.995)
    weight_decay = 5e-4
    num_workers = 0
    shuffle = True
    pin_memory = True
    test_after_epoch = 5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def configure_optimizer(self, model: IModel):
        if self.optimizer_name is OptimizerNames.SGD:
            return torch.optim.SGD(model.parameters(),
                                   self.lr)
        elif self.optimizer_name is OptimizerNames.SGDWithMomentum:
            return torch.optim.SGD(model.parameters(),
                                   self.lr,
                                   momentum=self.momentum)
        elif self.optimizer_name is OptimizerNames.RMSProp:
            return torch.optim.RMSprop(model.parameters(),
                                       self.lr,
                                       weight_decay=self.weight_decay,
                                       momentum=self.momentum)
        elif self.optimizer_name is OptimizerNames.Adam:
            return torch.optim.Adam(model.parameters(),
                                    self.lr,
                                    betas=self.betas,
                                    weight_decay=self.weight_decay)
        else:
            raise Exception(f"Invalid criterion name {self.criterion_name}!")

    def configure_criterion(self):
        if self.criterion_name is CriterionNames.BCELoss:
            return torch.nn.BCELoss()
        elif self.criterion_name is CriterionNames.CrossEntropyLoss:
            return torch.nn.CrossEntropyLoss()
        else:
            raise Exception(f"Invalid criterion name {self.criterion_name}!")
