import torch
from torch import nn
from pathlib import Path

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from helper_functions import accuracy_fn, eval_model


# Getting dataset
# Using MNIST dataset - FASHION MNIST
train_data = datasets.FashionMNIST(
    root="data", #where to store the data
    train=True, # do we want the training dataset
    download=True, # do we want to download the dataset
    transform=ToTensor(), # how do we want to transform the data
    target_transform=None # we don't need to transform the labels
)

test_data = datasets.FashionMNIST(
    root="data", #where to store the data
    train=False, # do we want the test dataset
    download=True, # do we want to download the dataset
    transform=ToTensor(), # how do we want to transform the data
    target_transform=None # we don't need to transform the labels
)


# prepare dataloader
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
class_names = train_data.classes

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

class fashionMNISTModel(nn.Module):
    """
        Model Architecture that replicates TinyVGG
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, # there is a trick to calculating this
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        # print(f"Output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"Output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"Output shape of classifier: {x.shape}")
        return x

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "vision_fashion_mnist_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
loaded_model = fashionMNISTModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

torch.manual_seed(42)

loaded_model_results = eval_model(
    model=loaded_model,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)

print(f"Loaded model results: {loaded_model_results}")
