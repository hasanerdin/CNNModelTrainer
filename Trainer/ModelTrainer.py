import logging
import os.path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.IModel import IModel
from Trainer.TrainConfigurator import TrainerConfig
from Models.ModelMetrics import ModelMetrics
from Models.ModelRunner import ModelRunner

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model: IModel, train_dataset, test_dataset, config: TrainerConfig):
        self.train_configurator = config

        self.train_dataloader = DataLoader(train_dataset,
                                           self.train_configurator.batch_size,
                                           self.train_configurator.shuffle,
                                           pin_memory=self.train_configurator.pin_memory,
                                           num_workers=self.train_configurator.num_workers)
        self.test_dataloader = DataLoader(test_dataset,
                                          self.train_configurator.batch_size,
                                          False,
                                          pin_memory=self.train_configurator.pin_memory,
                                          num_workers=self.train_configurator.num_workers)

        device = "cpu"
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            model = model.to(device)

        self.best_acc = 0
        self.start_epoch = 0
        if config.resume and os.path.exists("./checkpoint/ckpt.pth"):
            print(f"Model is being loaded...")
            checkpoint = torch.load("./checkpoint/ckpt.pth")
            model.load_state_dict(checkpoint["model"])
            self.best_acc = checkpoint["acc"]
            self.start_epoch = checkpoint["epoch"]

        optimizer = config.configure_optimizer(model)
        criterion = config.configure_criterion()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        self.model_runner = ModelRunner(model, optimizer, criterion, device)
        self.model_metrics = ModelMetrics()

    def save_checkpoint(self):
        logger.info(f"Saving model!")
        state = {
            "model": self.model_runner.model.state_dict(),
            "acc": self.model_metrics.test_acc[-1],
            "epoch": self.start_epoch + self.model_metrics.epoch
        }
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(state, "./checkpoint/ckpt.pth")

    def display_graphics(self):
        self.model_metrics.display_graphics()

    def show_image(self) -> None:
        images, labels = next(iter(self.train_dataloader))
        num_images = len(images)
        col_num = 6
        row_num = int(np.ceil(num_images / col_num))
        for image_index, image in enumerate(images):
            plt.subplot(row_num, col_num, image_index + 1)
            img = image.permute((1, 2, 0)).numpy()
            un_normalized_img = img / 2 + 0.5
            plt.imshow(un_normalized_img)
            plt.title(self.model_runner.model.class_names[labels[image_index]])
            plt.axis("off")

        plt.show()

    def check_test_results(self):
        if self.model_metrics.test_acc[-1] > self.best_acc:
            self.best_acc = self.model_metrics.test_acc[-1]
            self.save_checkpoint()

    def test(self) -> (float, float):
        test_acc, test_loss = self.model_runner.test(self.test_dataloader, self.model_metrics.epoch)
        self.model_metrics.add_test_results(test_acc, test_loss)
        self.check_test_results()
        return test_acc, test_loss

    def print_train_parameters(self):
        print(f"\tStart Epoch: {self.start_epoch}\n"
              f"\tBest Acc: {self.best_acc}\n"
              f"\tOptimizer: {self.train_configurator.optimizer_name}\n"
              f"\tLoss: {self.train_configurator.criterion_name}\n"
              f"\tEpochs: {self.train_configurator.epochs}\n"
              f"\tLR: {self.train_configurator.lr}")

    def train(self):
        print("Training is started!")
        self.print_train_parameters()
        print("******************************")
        for epoch in range(self.train_configurator.epochs):
            self.model_metrics.epoch = epoch
            train_acc, train_loss = self.model_runner.train(self.train_dataloader, epoch)
            self.model_metrics.add_train_results(train_acc, train_loss)

            if epoch % self.train_configurator.test_after_epoch == 0:
                self.test()
            self.scheduler.step()

        self.test()
        print("******************************\nTraining is ended!")
