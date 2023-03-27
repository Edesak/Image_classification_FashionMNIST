import torch
from torch import nn

def hello():
    print("Hello this is 03_models py file.")

class Fashion_modelv1(nn.Module):
    """
    This is model for chapter 03_computer vision with NonLinear layers
    """
    def __init__(self,
                 in_features:int,
                 hidden_units:int,
                 out_features:int):
        super().__init__()
        self.stacked_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=out_features),
            nn.ReLU()
        )
    def forward(self,x):
        return self.stacked_layers(x)

class Fashion_CNNv2(nn.Module):

    def __init__(self,
                 in_features:int,
                 hidden_units:int,
                 out_features:int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=out_features)
        )
    def forward(self,x):
        y = self.conv_block1(x)
        #print(y.shape)
        y = self.conv_block2(y)
        #print(y.shape)
        #print(self.conv_block2.parameters())
        y = self.classifier(y)
        return y

class baseline_model(nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )

    def forward(self,x):
        return self.layer_stack(x)







