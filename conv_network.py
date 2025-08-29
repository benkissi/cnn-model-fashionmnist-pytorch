import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

from helper_functions import accuracy_fn, print_train_time, train_step, test_step, eval_model, make_predictions, plot_conv_predictions

import random

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
# device = "cpu"

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

print(f"length of class_names: {len(class_names)}")
model = fashionMNISTModel(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

torch.manual_seed(42)

# images = torch.randn(size=(32, 3, 64, 64))
# test_image = images[0]

# print(f"test_image.shape: {test_image.shape}")

# conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0)
# conv_output = conv_layer(test_image)

# print(f"Conv layer output shape: {conv_output.shape}")
# # print(conv_output)

# max_pool = nn.MaxPool2d(kernel_size=2)
# max_pool_output = max_pool(conv_output)

# print(f"Max pool output shape: {max_pool_output.shape}")

# rand_image_tensor = torch.randn([1, 28, 28])
# print(f"Random image tensor shape: {rand_image_tensor.shape}")

# output = model(rand_image_tensor.unsqueeze(0).to(device))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 3

start_time = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(
        model=model,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )

    test_step(
        model=model,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

end_time = timer()
print_train_time(start_time, end_time, device=device)

model_results = eval_model(
    model=model,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device
)

print(f"Evaluation Results:\n....................\n{model_results}")

# random.seed(42)

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model, test_samples, device=device)

# convert pred probs to labels
pred_classes = pred_probs.argmax(dim=1)
print(f"Pred class: {pred_classes}, \nTrue labels: {test_labels}")

plot_conv_predictions(test_samples, test_labels, class_names, pred_classes)
