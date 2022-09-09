import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from model_config import image_size, h_dim, z_dim
from vae import VAE
import pandas as pd
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE_API:
    def __init__(self, model_path, dataset, num_reduced_dimensions=2, batch_size=128):
        self.model_path = model_path
        self.model = self.load_model(self.model_path)
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.num_reduced_dimensions = num_reduced_dimensions
        self.batch_size = batch_size


    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        model = VAE(image_size=image_size, h_dim=h_dim, z_dim=z_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


    def generate_scatterplot_dataframe(self):
        """creates dataframe containing low-dimensionality vectors and images
        that can be plotted on dashboard
        """
        z, x, labels = self.get_latent_vectors()
        z = [list([elem.item() for elem in i]) for i in z]

        for i in range(len(x)):
            image = x[i]
            save_image(image, os.path.join(os.getcwd(), f"assets/{i}.png"))

        df = pd.DataFrame({"id":[i for i in range(len(z))] , "z": z, "labels":labels})
        print(df)


    def reduce_dimensions(self):
        """run either t-SNE or PCA to reduce latent space to two dimensions"""
        pass


    def get_latent_vectors(self):
        """get latent vectors for <batch_size> images by running them into the model
        returns latent vectors, images, and associated labels
        """
        with torch.no_grad():
            x, labels = next(iter(self.dataloader))
            x_mod = x.to(device).view(-1, image_size)
            mu, log_var = self.model.encode(x_mod)
            z = self.model.reparameterize(mu, log_var)


            # save images to local directory
            return z, x, labels
    

    def generate_iterpolation_gif(self):
        """create interpolation_gif between two latent-space vectors"""
        pass