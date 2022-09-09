import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
import pandas as pd
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d

import numpy as np
import os
import shutil
import imageio

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
        print(epoch_number)
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
    
    
    def generate_image(self, latent_vector):
        latent_vector = torch.Tensor(latent_vector)
        with torch.no_grad():
            generated = self.model.decode(latent_vector).view(-1, 1, 28, 28)
            
            save_image(generated, os.path.join(os.getcwd(), f"assets/generated.png"))
        return "assets/generated.png"
            

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
    

    def stitch_images(self, directory, num_steps=100):
        """stitches images in specified directory to create a GIF
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        """
        filenames = [os.path.join(directory, f"{i}.png") for i in range(0, num_steps)]
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        assets_dir = os.path.join(os.getcwd(), 'assets/interpolation.gif')
        imageio.mimsave(assets_dir, images, duration=0.05)
        return 'assets/interpolation.gif' 


    def generate_iterpolation_gif(self, latent_vector1, latent_vector_2, num_steps=100):
        """create interpolation_gif between two latent-space vectors"""
        lv1 = np.array(latent_vector1)
        lv2 = np.array(latent_vector_2)
        linfit = interp1d([1,num_steps], np.vstack([lv1, lv2]), axis=0)

        # check if folder exists
        gif_dir = os.path.join(os.getcwd(), 'assets/gif_imgs/')
        if os.path.exists(gif_dir):
            shutil.rmtree(gif_dir)
        os.makedirs(gif_dir)
        # delete elements in folder if it exists

        latent_vectors =  linfit([i for i in range(1,num_steps+1)])
        latent_vectors = torch.Tensor(latent_vectors)

        with torch.no_grad():
            generated = self.model.decode(latent_vectors).view(-1, 1, 28, 28)
            for i in range(generated.shape[0]):
                new_gen = generated[i].view(-1, 1, 28, 28)
                save_image(new_gen, os.path.join(os.getcwd(), f"assets/gif_imgs/{i}.png"), normalize=True)
        
        return self.stitch_images(os.path.join(os.getcwd(), f"assets/gif_imgs/"))
        