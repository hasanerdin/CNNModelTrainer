from enum import Enum

from torchvision.datasets import MNIST, CIFAR10

from Trainer.TrainConfigurator import TrainerConfig
from Trainer.ModelTrainer import ModelTrainer
from Models.MNISTModel import MNISTModel
from Models.Cifar10Model import Cifar10NNModel

import Tranformations


class ModelNames(Enum):
    MNIST = 1
    Cifar10 = 2


def print_dataset_info(train_dataset, test_dataset):
    train_image_example = train_dataset[0][0].permute((1, 2, 0))
    print("*****************************")
    print(f"Data Info: \n"
          f"\tNumber of Train Data: {len(train_dataset)}\n"
          f"\tNumber of Test Data: {len(test_dataset)}\n"
          f"\tImage Size: {train_image_example.numpy().shape}")
    print("*****************************")


def get_model_trainer(model_name: ModelNames, config: TrainerConfig) -> ModelTrainer:
    def get_mnist_model_trainer():
        transform = Tranformations.mnist_transform

        mnist_train_set = MNIST(root="./data", train=True, transform=transform, download=True)
        mnist_test_set = MNIST(root="./data", train=False, transform=transform, download=True)
        print_dataset_info(mnist_train_set, mnist_test_set)

        return ModelTrainer(MNISTModel(), mnist_train_set, mnist_test_set, config)

    def get_cifar10_model_trainer():
        transform = Tranformations.cifar10_transform

        cifar10_train_set = CIFAR10(root="./data", train=True, transform=transform, download=True)
        cifar10_test_set = CIFAR10(root="./data", train=False, transform=transform, download=True)
        print_dataset_info(cifar10_train_set, cifar10_test_set)

        return ModelTrainer(Cifar10NNModel(), cifar10_train_set, cifar10_test_set, config)

    print(f"{model_name} Model is being prepared...")
    if model_name is ModelNames.MNIST:
        return get_mnist_model_trainer()
    elif model_name is ModelNames.Cifar10:
        return get_cifar10_model_trainer()
    else:
        raise Exception(f"Invalid Model Name {model_name}!")



