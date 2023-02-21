import torch
import torch.nn.functional as F
"""
While the former defines nn.Module classes, the latter uses a functional (stateless) approach.
To dig a bit deeper: nn.Modules are defined as Python classes and have attributes, e.g. a nn.Conv2d module will have some internal attributes like self.weight. F.conv2d however 
just defines the operation and needs all arguments to be passed (including the weights and bias
"""
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary