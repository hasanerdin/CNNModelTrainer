import torchvision.transforms as transforms

cifar10_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomRotation((-45, 45)),
     transforms.RandomHorizontalFlip()]
)

mnist_transform = transforms.ToTensor()
