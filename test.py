import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from vae import VAE
from vae_api import VAE_API

import plotly.express as px
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)


# model_path = "/Users/saahith/Desktop/variational-autoencoder/checkpoints/model_15.pt"
            #    '/Users/saahith/Desktop/variational-autoencodercheckpoints/model_15.pt'
vae_api = VAE_API(os.getcwd()+"/checkpoints/", dataset, batch_size=512)
# latent_vectors, x, labels = vae_api.get_latent_vectors()

df = vae_api.generate_scatterplot_dataframe(epoch_number=15)

fig = px.scatter(df, x="x", y="y", color="labels")
fig.show()

print(df)

# print(latent_vectors.shape)
# print(labels.shape)
