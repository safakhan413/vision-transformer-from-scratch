import torch
import torch.nn.functional as F
"""
While the former defines nn.Module classes, the latter uses a functional (stateless) approach.
To dig a bit deeper: nn.Modules are defined as Python classes and have attributes, e.g. a nn.Conv2d module will have some internal attributes like self.weight. F.conv2d however 
just defines the operation and needs all arguments to be passed (including the weights and bias
Steps:
- Data
- Patches Embeddings
    - CLS Token
    - Position Embedding
- Transformer
    - Attention
    - Residuals
    - MLP
    - TransformerEncoder
- Head
- ViT
"""
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
"""
to resize and crop images check this https://realpython.com/image-processing-with-the-python-pillow-library/#:~:text=PIL%20stands%20for%20Python%20Imaging,includes%20support%20for%20Python%203.

Python Imaging Library is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats. 
"""
from torchvision.transforms import Compose, Resize, ToTensor
"""
Transforms are common image transformations available in the torchvision.transforms module. They can be chained together using Compose.
The transformations that accept tensor images also accept batches of tensor images. A Tensor Image is a tensor with (C, H, W) shape, 
where C is a number of channels, H and W are image height and width. A batch of Tensor Images is a tensor of (B, C, H, W) shape, where B is a number of images in the batch.
"""
from einops import rearrange, reduce, repeat
"""
y = x.transpose(0, 2, 3, 1)
We write comprehensible code

y = rearrange(x, 'b c h w -> b h w c')
einops supports widely used tensor packages (such as numpy, pytorch, chainer, gluon, tensorflow), and extends them.

# or compose a new dimension of batch and width
rearrange(ims, 'b h w c -> h (b w) c') ## concats all images horizontally or width wise

https://einops.rocks/1-einops-basics/ very useful library to handle dimensions

# average over batch
reduce(ims, 'b h w c -> h w c', 'mean')

"""
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
"""
from torchvision import models
from torchsummary import summary

vgg = models.vgg16()
summary(vgg, (3, 224, 224))

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
           Linear-32                 [-1, 4096]     102,764,544
             ReLU-33                 [-1, 4096]               0
          Dropout-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]      16,781,312
             ReLU-36                 [-1, 4096]               0
          Dropout-37                 [-1, 4096]               0
           Linear-38                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.59
Params size (MB): 527.79
Estimated Total Size (MB): 746.96
"""

# Download pizza, steak, sushi images from GitHub
from download_data import *
################################### This part os for dlownloading the data. Once download is completed, comment it out #################

# image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
#                            destination="pizza_steak_sushi")
# print(image_path)

# # Setup directory paths to train and test images
# train_dir = image_path / "train"
# test_dir = image_path / "test"

# Now we've got some data, let's now turn it into DataLoader's.


from data_setup import *

# Create image size (from Table 3 in the ViT paper) 
IMG_SIZE = 224

# Composes several transforms together. This transform does not support torchscript
# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])           
print(f"Manually created transforms: {manual_transforms}")

# Set the batch size
BATCH_SIZE = 10 # this is lower than the ViT paper but it's because we're starting small
# image_path_new = r"S:code\data\pizza_steak_sushi"
# train_dir_downloaded = image_path_new + "\train"
# test_dir_downloaded = image_path_new + "\test"
train_dir_downloaded = "data/pizza_steak_sushi/train"
test_dir_downloaded = "data/pizza_steak_sushi/test"
# Create data loaderscl
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir_downloaded,
    test_dir=train_dir_downloaded,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

print(train_dataloader)
if __name__ == '__main__':
  # Get a batch of images
  image_batch, label_batch = next(iter(train_dataloader))

  # Get a single image from the batch
  image, label = image_batch[0], label_batch[0]

  # View the batch shapes
  print(image.shape, label)
  # output torch.Size([3, 224, 224]) tensor(2). As can be seen color_channels is first dim of image tensor

  # Plot image with matplotlib
  plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
  plt.title(class_names[label])
  plt.axis(False)
  plt.show() # Add this line to actually see the plot

  
  ################### Replication of ViT paper #######################################

# So our model inputs are: images of pizza, steak and sushi.

# And our ideal model outputs are: predicted labels of pizza, steak or sushi.
# Figure 1, Table 3.1 containing Mathematical equations, Table 1 showing hyperparameter settings will be our guiding light to create the ViT architecture to solve our problem

# Steps in Figure 1 Explained:
# 
# Patch + Position Embedding (inputs) - Turns the input image into a sequence of image patches and add a position number what order the patch comes in.

# Linear projection of flattened patches (Embedded Patches) - The image patches get turned into an embedding, the benefit of using an embedding rather 
#   than just the image values is that an embedding is a learnable representation (typically in the form of a vector) of the image that can improve with training.

# Norm - This is short for "Layer Normalization" or "LayerNorm", a technique for regularizing (reducing overfitting) a neural network, you can use 
#   LayerNorm via the PyTorch layer torch.nn.LayerNorm().

# Multi-Head Attention - This is a Multi-Headed Self-Attention layer or "MSA" for short. You can create an MSA layer via the PyTorch layer torch.nn.MultiheadAttention().
#   Image is broken down into patches rather than performing Self Attention on each of the layers so that computation cost

# MLP (or Multilayer perceptron) - A MLP can often refer to any collection of feedforward layers (or in PyTorch's case, a collection of layers with a forward() method). 
#   In the ViT Paper, the authors refer to the MLP as "MLP block" and it contains two torch.nn.Linear() layers with a torch.nn.GELU() non-linearity activation
#   in between them (section 3.1) and a torch.nn.Dropout() layer after each (Appendex B.1).

# Transformer Encoder - The Transformer Encoder, is a collection of the layers listed above. There are two skip connections inside the Transformer encoder
#   (the "+" symbols) meaning the layer's inputs are fed directly to immediate layers as well as subsequent layers. The overall ViT architecture is
#   comprised of a number of Transformer encoders stacked on top of eachother.

# MLP Head - This is the output layer of the architecture, it converts the learned features of an input to a class output. Since we're working on image classification, you could also call this the "classifier head". The structure of the MLP Head is similar to the MLP block. 