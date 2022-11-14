import matplotlib.pyplot as plt


class ModelMetrics:
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    epoch = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def add_train_results(self, train_acc: float, train_loss: float) -> None:
        self.train_acc.append(train_acc)
        self.train_losses.append(train_loss)

    def add_test_results(self, test_acc: float, test_loss: float) -> None:
        extend_num = len(self.train_acc) - len(self.test_acc)
        self.test_acc.extend([test_acc for _ in range(extend_num)])
        self.test_losses.extend([test_loss for _ in range(extend_num)])

    def display_graphics(self):
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epoch + 1), self.train_acc, "b", label="Train Acc")
        plt.plot(range(self.epoch + 1), self.test_acc, "r", label="Test Acc")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(self.epoch + 1), self.train_losses, "b", label="Train Loss")
        plt.plot(range(self.epoch + 1), self.test_losses, "r", label="Test Loss")
        plt.legend()

        plt.show()
