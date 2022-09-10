from asyncore import read
from tkinter import LAST
from tkinter.tix import IMAGE
import dash
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

import dash_bootstrap_components as dbc



IMAGE_1 = None
IMAGE_2 = None

df = vae_api.generate_scatterplot_dataframe(num_epochs)
GRAPH = px.scatter(df, x="x", y="y", color="labels", title="tSNE embeddings of latent space vectors", template="plotly_dark")



RADIO_BUTTONS = ["Image 1", "Image 2"]
INITIAL_RADIO_SELECTION = RADIO_BUTTONS[0]
RADIO_SELECTION = RADIO_BUTTONS[0]



####
## Variables for latent space visualization
LATENT_SPACE_DIM = z_dim


LAST_SELECTED_VECTOR = [0,0,0,0,0,0,0,0,0,0]
LAST_RECONSTRUCTEd_VECTOR = None
latent_vector_1 = None
latent_vector_2 = None


# app = Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

# create assets dir if it doesn't already exist
assets_directory = os.path.join(os.getcwd(), "assets/")
if not os.path.exists(assets_directory):
    os.makedirs(assets_directory)

latent_space_inputs = [html.H2("Reconstruction Vector")]
for i in range(LATENT_SPACE_DIM):
    latent_space_inputs.append(
        dcc.Input(id=f"latent_input_{i}", type = "number", placeholder = f"x{i}", value="0", readOnly=True, style={"color": "#ffffff", "background-color": "#080000"}))#}))
    latent_space_inputs.append(html.Br())

latent_space_inputs.append(
    html.Button("Reconstruct last selected latent vector", id = "copy_latent_vector")
)
latent_space_inputs.append(html.Br())

app.layout = html.Div([
    
    

    html.H1("VAE Visualizer", style={"text-align":"center"}),
    dcc.Graph(id="scatter-plot", figure=GRAPH),


    html.Div([

    html.Div([


    # latent-space interpolation div

    html.Div([
    dcc.RadioItems(RADIO_BUTTONS, INITIAL_RADIO_SELECTION, id="radio_button", inputStyle ={"margin-left":"50px"}),

    html.H4(f"Image Being Edited: {INITIAL_RADIO_SELECTION}", id="radio_info"),
    
    html.H4( f"{RADIO_BUTTONS[0]}", id="point1"),
    html.Img(src='', id="image1", height=200, width=200),
    html.H4(f"{RADIO_BUTTONS[1]}", id="point2"),
    html.Img(src="", id="image2", height=200, width=200), html.Br(),
    html.Button("Interpolate", id = "interpolate"),
    ], style={'padding': 10, 'flex': 1}),

    # separate div for the actual gif
    html.Div([
        html.H2("Interpolation GIF"),
        html.Img(src="", id="interpolation-gif", height=300, width=300, style={"vertical-align": "middle"})], style={'padding': 10, 'flex': 1})
    ], style={'padding': 10, 'flex': 1, 'display': 'flex', 'flex-direction': 'row'}),
    
    
    
    # latent-space reconstruction div
    html.Div([html.Div(latent_space_inputs, style={'padding': 10, 'flex': 1}), html.Div([html.H2("Reconstructed Image"), html.Img(src='', id="generated_img", height=300, width=300)], style={'padding': 10, 'flex': 1})], style={'padding': 10, 'flex': 1, 'display': 'flex', 'flex-direction': 'row'})
    


    ], style={'display': 'flex', 'flex-direction': 'row'}) 

])



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


@app.callback(
    Output("interpolation-gif", "src"),
    # Input("num-steps", "value"),
    Input("interpolate", "n_clicks"))
def get_interpolation_gif(n_clicks):
    global latent_vector_1, latent_vector_2
    if latent_vector_1 == None or latent_vector_2 == None:
        return ""
    else:
        # print(latent_vector_1)
        # print(latent_vector_2)
        return vae_api.generate_iterpolation_gif(latent_vector1=latent_vector_1, latent_vector_2=latent_vector_2, num_steps=100)



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
    Output('generated_img', 'src'),
    Input('copy_latent_vector', 'n_clicks'),
)
def update_latent_vector(n_clicks):
    global LAST_SELECTED_VECTOR
    if LAST_SELECTED_VECTOR == None:
        LAST_SELECTED_VECTOR = [0]*LATENT_SPACE_DIM
    return LAST_SELECTED_VECTOR + [vae_api.generate_image(LAST_SELECTED_VECTOR)]


# choose point on scatterplot for latent-space interpolation
@app.callback(
    Output('image1', 'src'),
    Output('image2', 'src'),
    Input('scatter-plot', 'clickData'))
def display_click_data(clickData):
    print("callback4")
    global IMAGE_1, IMAGE_2, RADIO_SELECTION, df, LAST_SELECTED_VECTOR, latent_vector_1, latent_vector_2
    # print(clickData)
    # print(RADIO_SELECTION)
    if clickData == None:
        return IMAGE_1, IMAGE_2

    # print(clickData["points"])
    if RADIO_SELECTION == RADIO_BUTTONS[0]:
        # print("editing data point 1")
        # IMAGE_1 = os.path.join(os.getcwd(), f'assets/{clickData["points"][0]["pointIndex"]}.png')
        index = clickData["points"][0]["pointIndex"]
        IMAGE_1 = f'./assets/{clickData["points"][0]["pointIndex"]}.png'
        LAST_SELECTED_VECTOR = df.loc[index]["z"]
        latent_vector_1 = LAST_SELECTED_VECTOR


    else:
        IMAGE_2 = os.path.join(os.getcwd(), f'assets/{clickData["points"][0]["pointIndex"]}.png')
        index = clickData["points"][0]["pointIndex"]
        IMAGE_2 = f'./assets/{clickData["points"][0]["pointIndex"]}.png'
        LAST_SELECTED_VECTOR = df.loc[index]["z"]
        latent_vector_2 = LAST_SELECTED_VECTOR


    return IMAGE_1, IMAGE_2

if __name__ == "__main__":
    app.run_server(debug=True)
