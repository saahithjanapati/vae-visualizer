import torch
from torchvision.utils import save_image

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import os

from model_config import image_size, h_dim, z_dim
from vae import VAE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE_API:
    def __init__(self, checkpoint_dir, dataset, num_reduced_dimensions=2, batch_size=128):
        self.checkpoint_dir = checkpoint_dir
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.num_reduced_dimensions = num_reduced_dimensions
        self.batch_size = batch_size
        self.model = None


    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model = VAE(image_size=image_size, h_dim=h_dim, z_dim=z_dim).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


    def generate_scatterplot_dataframe(self, epoch_number=15):
        """creates dataframe containing low-dimensionality vectors and images
        that can be plotted on dashboard
        """
        # load model for specified epoch
        self.load_model(os.path.join(self.checkpoint_dir, f"model_{epoch_number}.pt"))
        
        z, x, labels = self.get_latent_vectors()
        z = [list([elem.item() for elem in i]) for i in z]
        
        #TSNE for dimension reduction
        reduced_latent_vectors = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(np.array(z))
        x_coords, y_coords = [elem[0] for elem in reduced_latent_vectors], [elem[1] for elem in reduced_latent_vectors]

        for i in range(len(x)):
            image = x[i]
            save_image(image, os.path.join(os.getcwd(), f"assets/{i}.png"))

        df = pd.DataFrame({"id":[i for i in range(len(z))],"x":x_coords, "y":y_coords, "z": z, "labels":labels})
        return df


    def get_latent_vectors(self):
        """get latent vectors for <batch_size> images by running them into the model
        returns latent vectors, images, and associated labels
        """
        with torch.no_grad():
            x, labels = next(iter(self.dataloader))
            x_mod = x.to(device).view(-1, image_size)
            mu, log_var = self.model.encode(x_mod)
            z = self.model.reparameterize(mu, log_var)
            return z, x, labels
    

    def generate_iterpolation_gif(self):
        """create interpolation_gif between two latent-space vectors"""
        pass