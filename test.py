import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from vae import VAE
from vae_api import VAE_API


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)


model_path = "/Users/saahith/Desktop/variational-autoencoder/checkpoints/model_15.pt"
vae_api = VAE_API(model_path, dataset)
x = vae_api.get_latent_vectors()
