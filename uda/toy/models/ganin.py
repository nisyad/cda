import torch.nn as nn
from torch.autograd import Function
from torchinfo import summary


class GaninModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            # nn.Dropout2d(),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, input_data, alpha=0):

        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseGradientLayer.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output


class ReverseGradientLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        return -ctx.alpha * grad_output, None


class SimpleClassifier(nn.Module):
    """Simple classifier based on Ganin et al. architecture. It does not include
       discriminator.
    """
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

    def forward(self, input_data):

        feature = self.feature(input_data)
        feature = feature.reshape(-1, 48 * 4 * 4)
        return self.classifier(feature)


if __name__ == "__main__":
    model = GaninModel()
    summary(model, (3, 28, 28))
