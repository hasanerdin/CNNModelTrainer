from tqdm import tqdm
import torch

from Models.IModel import IModel


class ModelRunner:
    def __init__(self, model: IModel, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    @staticmethod
    def calculate_accuracy(predictions, labels):
        predicted = torch.argmax(predictions, dim=1)
        acc = predicted.eq(labels).sum().item() / len(labels)
        return acc

    @staticmethod
    def mean(batch_results: list) -> float:
        return sum(batch_results) / len(batch_results)

    def train(self, train_dataloader, epoch: int) -> (float, float):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        batch_acc = []
        batch_loss = []
        for batch_index, (batch_samples, batch_labels) in pbar:
            batch_samples = batch_samples.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(batch_samples)
            acc = self.calculate_accuracy(predictions, batch_labels)
            batch_acc.append(acc)

            loss = self.criterion(predictions, batch_labels)
            batch_loss.append(loss.mean().item())

            loss.backward()
            self.optimizer.step()

            pbar.set_description(f"Epoch {epoch + 1} => "
                                 f"Train Loss: {self.mean(batch_loss): .4f}, "
                                 f"Train Acc: {self.mean(batch_acc): .4f}")

        return self.mean(batch_acc), self.mean(batch_loss)

    def test(self, test_dataloader, epoch: int) -> (float, float):
        with torch.no_grad():
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

            batch_acc = []
            batch_loss = []
            for batch_index, (batch_samples, batch_labels) in pbar:
                batch_samples = batch_samples.to(self.device)
                batch_labels = batch_labels.to(self.device)

                predictions = self.model(batch_samples)
                acc = self.calculate_accuracy(predictions, batch_labels)
                batch_acc.append(acc)

                loss = self.criterion(predictions, batch_labels)
                batch_loss.append(loss.mean().item())

                pbar.set_description(f"Epoch {epoch + 1} => "
                                     f"Test Loss: {self.mean(batch_loss): .4f}, "
                                     f"Test Acc: {self.mean(batch_acc): .4f}")

            return self.mean(batch_acc), self.mean(batch_loss)
