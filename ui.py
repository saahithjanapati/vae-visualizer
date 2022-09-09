from tkinter import LAST
from tkinter.tix import IMAGE
from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px


from vae_api import VAE_API

import torch
import torchvision
from torchvision import transforms
from vae_api import VAE_API
from model_config import num_epochs, z_dim

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

vae_api = VAE_API(os.path.join(os.getcwd(), "checkpoints/"), dataset, batch_size=512)


EPOCH_SLIDER_MIN = 1
EPOCH_SLIDER_MAX = num_epochs
EPOCH_SLIDER_INITIAL_VALUE = 5


IMAGE_1 = None
IMAGE_2 = None


# GRAPH_DATA_X = [0, 1, 2, 3, 4]
# GRAPH_DATA_Y = [x**EPOCH_SLIDER_INITIAL_VALUE for x in GRAPH_DATA_X]
df = vae_api.generate_scatterplot_dataframe(num_epochs)
GRAPH = px.scatter(df, x="x", y="y", color="labels", title="tSNE embeddings of latent space vectors")



RADIO_BUTTONS = ["Image 1", "Image 2"]
INITIAL_RADIO_SELECTION = RADIO_BUTTONS[0]
RADIO_SELECTION = RADIO_BUTTONS[0]



####
## Variables for latent space visualization
LATENT_SPACE_DIM = z_dim
# LATENT_SPACE_MIN = 0
# LATENT_SPACE_MAX = 10
# LATENT_SPACE_SLIDER_STEP = 0.1
# LATENT_SPACE_SLIDER_INITIAL_VALUE = (LATENT_SPACE_MIN + LATENT_SPACE_MAX)/2
# LATENT_SPACE_MARKS = {i: str(i) for i in range(EPOCH_SLIDER_MIN, EPOCH_SLIDER_MAX+1)}


LAST_SELECTED_VECTOR = None



app = Dash(__name__)

# latent_space_sliders = [
#         dcc.Slider(min=LATENT_SPACE_MIN,max=LATENT_SPACE_MAX, step=LATENT_SPACE_SLIDER_STEP, 
#         marks=LATENT_SPACE_MARKS, 
#         value=LATENT_SPACE_SLIDER_INITIAL_VALUE, id=f"latent-space-dim-{i}") for i in range(LATENT_SPACE_DIM)]



latent_space_inputs = [
    dcc.Input(
        id=f"latent_input_{i}",
        type = "number",
        placeholder = f"x{i}",
        value="0"
    )
    for i in range(LATENT_SPACE_DIM)]

latent_space_inputs.append(
    html.Button("Copy last selected latent vector", id = "copy_latent_vector")
)

app.layout = html.Div([
    html.H1("VAE Visualizer", style={"text-align":"center"}),

    dcc.Graph(id="scatter-plot", figure=GRAPH),
    # dcc.Slider(
    #     min=EPOCH_SLIDER_MIN,
    #     max=EPOCH_SLIDER_MAX,
    #     step=None,
    #     marks={i: str(i) for i in range(EPOCH_SLIDER_MIN, EPOCH_SLIDER_MAX+1)},
    #     value=EPOCH_SLIDER_INITIAL_VALUE, id="epoch-slider"),
    # html.H6(f"Epoch number: {EPOCH_SLIDER_INITIAL_VALUE}", id="epoch-label"),
    dcc.RadioItems(RADIO_BUTTONS, INITIAL_RADIO_SELECTION, id="radio_button"),
    html.H6(f"Image Being Edited: {INITIAL_RADIO_SELECTION}", id="radio_info"),
    
    html.H6( f"{RADIO_BUTTONS[0]}", id="point1"),
    html.Img(src='', id="image1", height=100, width=100),
    html.H6(f"{RADIO_BUTTONS[1]}", id="point2"),
    html.Img(src="", id="image2", height=100, width=100), html.Br(), 
    html.Button("Interpolate", id = "interpolate"),    
    html.Div(latent_space_inputs),
    html.Img(src='', id="generated_img", height=100, width=100)])




# html.H6(f"{RADIO_BUTTONS[1]} Value: {DATA_POINT_2}", id="point2")] + latent_space_sliders)



# moidfy scatterplot graph to display individual latent space vector
# individual callback for each input :(

# def update_latent_vector(dim=0):
    

# for latent_dim in range(LATENT_SPACE_DIM):
#     @app.callback(
#         Output()
#     )




# change point selection radio button (for latent-space interpolation gifs)
@app.callback(
    Output("radio_info", "children"),
    Input("radio_button", "value"))
def update_radio_selection(option):
    print("callback1")
    global RADIO_SELECTION
    RADIO_SELECTION = option
    print(f"RADIO_SELECTION: {RADIO_SELECTION}")
    return f"Image Being Edited: {RADIO_SELECTION}"


# @app.callback(
#     Input("radio_button", "value"))
# def update_radio_selection(option):
#     global RADIO_SELECTION
#     RADIO_SELECTION = option

# select different epoch to visualize with the slider
# @app.callback(
#     Output("epoch-label", "children"),
#     Input("epoch-slider", "value"))
# def update_epoch_number(epoch_number):
#     print("callback2")
#     return f"Epoch number: {epoch_number}" 



# update graph with new epoch number
# @app.callback(
#     Output("scatter-plot", "figure"),
#     Input("epoch-slider", "value"))
# def update_graph(epoch_number):
#     print("callback3")
#     # print(f"hello{epoch_number}")
#     global GRAPH
#     df = vae_api.generate_scatterplot_dataframe(epoch_number)
#     GRAPH = px.scatter(df, x="x", y="y", color="labels")
#     return GRAPH


@app.callback(
    Output("latent_input_0", "value"),
    Output("latent_input_1", "value"),
    Output("latent_input_2", "value"),
    Output("latent_input_3", "value"),
    Output("latent_input_4", "value"),
    Output("latent_input_5", "value"),
    Output("latent_input_6", "value"),
    Output("latent_input_7", "value"),
    Output("latent_input_8", "value"),
    Output("latent_input_9", "value"),
    Input('copy_latent_vector', 'n_clicks'),
)
def update_latent_vector(n_clicks):
    if LAST_SELECTED_VECTOR == None:
        return [0]*LATENT_SPACE_DIM
    return LAST_SELECTED_VECTOR


@app.callback(
    Output('generated_img', 'src'),
    Input("latent_input_0", "value"),
    Input("latent_input_1", "value"),
    Input("latent_input_2", "value"),
    Input("latent_input_3", "value"),
    Input("latent_input_4", "value"),
    Input("latent_input_5", "value"),
    Input("latent_input_6", "value"),
    Input("latent_input_7", "value"),
    Input("latent_input_8", "value"),
    Input("latent_input_9", "value")
)
def generate_new_image(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    global LAST_SELECTED_VECTOR
    LAST_SELECTED_VECTOR = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    img_path = vae_api.generate_image(LAST_SELECTED_VECTOR)
    return img_path


# choose point on scatterplot for latent-space interpolation
@app.callback(
    Output('image1', 'src'),
    Output('image2', 'src'),
    Input('scatter-plot', 'clickData'))
def display_click_data(clickData):
    print("callback4")
    global IMAGE_1, IMAGE_2, RADIO_SELECTION, df, LAST_SELECTED_VECTOR
    # print(clickData)
    # print(RADIO_SELECTION)
    if clickData == None:
        return IMAGE_1, IMAGE_2

    # print(clickData["points"])
    if RADIO_SELECTION == RADIO_BUTTONS[0]:
        # print("editing data point 1")
        IMAGE_1 = os.path.join(os.getcwd(), f'assets/{clickData["points"][0]["pointIndex"]}.png')
        index = clickData["points"][0]["pointIndex"]
        IMAGE_1 = f'./assets/{clickData["points"][0]["pointIndex"]}.png'
        LAST_SELECTED_VECTOR = df.loc[index]["z"]



    else:
        IMAGE_2 = os.path.join(os.getcwd(), f'assets/{clickData["points"][0]["pointIndex"]}.png')
        index = clickData["points"][0]["pointIndex"]

        IMAGE_2 = f'./assets/{clickData["points"][0]["pointIndex"]}.png'
        LAST_SELECTED_VECTOR = df.loc[index]["z"]
    
    print(LAST_SELECTED_VECTOR)


    return IMAGE_1, IMAGE_2



# <img id="image1" src="/Users/saahith/Desktop/variational-autoencoder/assets/201.png">




# update graph according to epoch slider value
# @app.callback(
#     Output("scatter-plot", "figure"),
#     Input("epoch-slider", "value"))
# def update_bar_chart(slider_range):
#     fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
#     return fig






if __name__ == "__main__":
    app.run_server(debug=True)
    




