import random
import torch
import numpy as np
from argparse import ArgumentParser

from Trainer.TrainConfigurator import OptimizerNames, CriterionNames
from Trainer.TrainerCreator import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float, help="Learning Rate")
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs")
    parser.add_argument("--model_name", default=1, type=int, choices=range(1, len(ModelNames) + 1), help="Model Name")
    parser.add_argument("--optim", default=1, type=int, choices=range(1, len(OptimizerNames) + 1), help="Optimizer")
    parser.add_argument("--loss", default=2, type=int, choices=range(1, len(CriterionNames) + 1), help="Loss Function")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight Decay")
    parser.add_argument("--resume", default=False, type=bool, help="Run with saved model")

    return parser.parse_args()


def seed(seed_id):
    torch.manual_seed(seed_id)
    np.random.seed(seed_id)
    random.seed(seed_id)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(123)

    args = parse_args()

    train_config = TrainerConfig(epochs=args.epochs,
                                 optimizer_name=OptimizerNames(args.optim),
                                 criterion_name=CriterionNames(args.loss),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 resume=args.resume)

    model_trainer = get_model_trainer(ModelNames(args.model_name), train_config)
    model_trainer.train()
    model_trainer.display_graphics()
