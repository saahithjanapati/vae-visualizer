# vae-visualizer   <img src="https://github.com/saahithjanapati/vae-visualizer/blob/main/images/42.gif" width="50" height="50"/>


![Screenshot](https://github.com/saahithjanapati/vae-visualizer/blob/main/images/screenshot.png)


￼
￼



Images from: https://lilianweng.github.io/posts/2018-08-12-vae/


## How to run:
This tool was built and tested with Python 3.8.1. First, install the necessary requirements using:

```pip instal requirements.txt```


Then, start the UI by running:
```python ui.py```

This will start a server on your localhost. Navigate to the outputted link using your browser. 



## What is this?

*If you are unfamiliar with Variational Autoencoders, read the Background section below.*

This tool lets you step inside the mind of a variational autoencoder trained on the MNIST dataset. 


### Graph:
The graph at the top of the screen is a 2D representation of the 10-dimensional space that our network compresses data into. To create this graph, we select 512 MNIST images. We then run these images through the encoder network and sample 10-dimensional representations for each image. Using the t-SNE dimensionality reduction algorithm, we reduce dimensionality of this data to 2 dimensions so we can depict it on our graph. The data are color-coded according to their class (all 0’s will be one color, all 1’s will be another color, etc.).


### Interpolation

The bottom left of the UI is for creating a GIF transitioning from one number to another. This achieved by interpolating vectors between the sampled latent vectors representing the two images. The two images can be selected by using the toggle button and clicking points on the graph described above. When the ‘Interpolate’ button is pressed, a GIF of 100 frames is created by running each of the interpolated vectors through the decoder network and stitching these frames together.


### Reconstruction
The bottom right-hand side of the screen represents depicts a reconstruction of the last-selected image using its sampled 10-d latent vector representation. The resemblance of this reconstructed image to the original image is an indicator of how accurate the compression of information is.









## Background:
Autoencoders are neural networks that learn how to compress their training data.  These models are composed of two parts, the encoder and the decoder. The encoder compresses the original data to a vector. The decoder the takes this compressed representation and tries to regenerate the original data. 

￼![Screenshot](https://github.com/saahithjanapati/vae-visualizer/blob/main/images/autoencoder.png)


Variational autoencoders are a spin on this model that their training data with probability distributions instead of plain vectors. The decoder then samples a vector from this probability distribution, and tries to regenerate the original data with this vector. It turns out that this helps the network learn better compressed representations.

￼￼![Screenshot](https://github.com/saahithjanapati/vae-visualizer/blob/main/images/variationalautoencoder.png)

Variational Autoencoder




## Acknowledgements:
Images in the Background section were taken from [this blog post](https://lilianweng.github.io/posts/2018-08-12-vae/).
The code in `vae.py` and `train.py` is from [Yunjey Choi's PyTorch tutorial repository](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/variational_autoencoder/main.py). All other code is my own, unless otherwise noted.

