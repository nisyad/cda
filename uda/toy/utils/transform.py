from torchvision import transforms


def get_transform(dataset):
    if dataset == "mnist":
        return transforms.Compose([transforms.Normalize((0.5, ), (0.5, ))])

    elif dataset == "mnist_m":
        return transforms.Compose(
            [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    elif dataset == "svhn":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(28),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    elif dataset == 'usps':
        return transforms.Compose([transforms.Resize(28)])
