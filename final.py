from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px


# TODO: make a thing to select two points on the scatter plot
# TODO: make a gif maker class
# TODO: make slider worrk one-way
# TODO: integrate with model
# TODO: plain latent space explorer, plot on scatterplot with different color (just adjust vector values and see what happens ???)


EPOCH_SLIDER_MIN = 0
EPOCH_SLIDER_MAX = 9
EPOCH_SLIDER_INITIAL_VALUE = 3


DATA_POINT_1 = None
DATA_POINT_2 = None


GRAPH_DATA_X = [0, 1, 2, 3, 4]
GRAPH_DATA_Y = [x**EPOCH_SLIDER_INITIAL_VALUE for x in GRAPH_DATA_X]
GRAPH = px.scatter(x=GRAPH_DATA_X, y=GRAPH_DATA_Y)


RADIO_BUTTONS = ["Data Point 1", "Data Point 2"]
INITIAL_RADIO_SELECTION = RADIO_BUTTONS[0]
RADIO_SELECTION = RADIO_BUTTONS[0]



####
## Variables for latent space visualization
LATENT_SPACE_DIM = 2
LATENT_SPACE_MIN = 0
LATENT_SPACE_MAX = 10
LATENT_SPACE_SLIDER_STEP = 0.1
LATENT_SPACE_SLIDER_INITIAL_VALUE = (LATENT_SPACE_MIN + LATENT_SPACE_MAX)/2
LATENT_SPACE_MARKS = {i: str(i) for i in range(EPOCH_SLIDER_MIN, EPOCH_SLIDER_MAX+1)}


app = Dash(__name__)

latent_space_sliders = [
        dcc.Slider(min=LATENT_SPACE_MIN,max=LATENT_SPACE_MAX, step=LATENT_SPACE_SLIDER_STEP, 
        marks=LATENT_SPACE_MARKS, 
        value=LATENT_SPACE_SLIDER_INITIAL_VALUE, id=f"latent-space-dim-{i}") for i in range(LATENT_SPACE_DIM)]


app.layout = html.Div([
    dcc.Graph(id="scatter-plot", figure=GRAPH),
    dcc.Slider(
        min=EPOCH_SLIDER_MIN,
        max=EPOCH_SLIDER_MAX,
        step=None,
        marks={i: str(i) for i in range(EPOCH_SLIDER_MIN, EPOCH_SLIDER_MAX)},
        value=EPOCH_SLIDER_INITIAL_VALUE, id="epoch-slider"),
    html.H6(f"Epoch number: {EPOCH_SLIDER_INITIAL_VALUE}", id="epoch-label"),
    dcc.RadioItems(RADIO_BUTTONS, INITIAL_RADIO_SELECTION, id="radio_button"),
    html.H6(f"Data Point Being Edited: {INITIAL_RADIO_SELECTION}", id="radio_info"),
    
    html.H6( f"{RADIO_BUTTONS[0]} Value: {DATA_POINT_1}", id="point1"),
    html.H6(f"{RADIO_BUTTONS[1]} Value: {DATA_POINT_2}", id="point2")] + latent_space_sliders)



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
    global RADIO_SELECTION
    RADIO_SELECTION = option
    print(f"RADIO_SELECTION: {RADIO_SELECTION}")
    return f"Data Point Being Edited: {RADIO_SELECTION}"



# select different epoch to visualize with the slider
@app.callback(
    Output("epoch-label", "children"),
    Input("epoch-slider", "value"))
def update_epoch_number(epoch_number):
    return f"Epoch number: {epoch_number}" 



# update graph with new epoch number
@app.callback(
    Output("scatter-plot", "figure"),
    Input("epoch-slider", "value"))
def update_graph(epoch_number):
    global GRAPH_DATA_X, GRAPH_DATA_Y, GRAPH
    GRAPH_DATA_X = [0, 1, 2, 3, 4]
    GRAPH_DATA_Y = [x**epoch_number for x in GRAPH_DATA_X]
    GRAPH = px.scatter(x=GRAPH_DATA_X, y=GRAPH_DATA_Y)
    return GRAPH


# choose point on scatterplot for latent-space interpolation
@app.callback(
    Output('point1', 'children'),
    Output('point2', 'children'),
    Input('scatter-plot', 'clickData'))
def display_click_data(clickData):
    global DATA_POINT_1, DATA_POINT_2, RADIO_SELECTION
    print(clickData)
    print(RADIO_SELECTION)
    if clickData == None:
        return str(DATA_POINT_1), str(DATA_POINT_2)
    if RADIO_SELECTION == RADIO_BUTTONS[0]:
        print("editing data point 1")
        DATA_POINT_1 = (clickData["points"][0]["x"], clickData["points"][0]["y"])
    else:
        DATA_POINT_2 = (clickData["points"][0]["x"], clickData["points"][0]["y"])
    return str(DATA_POINT_1), str(DATA_POINT_2)





# update graph according to epoch slider value
# @app.callback(
#     Output("scatter-plot", "figure"),
#     Input("epoch-slider", "value"))
# def update_bar_chart(slider_range):
#     fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
#     return fig






if __name__ == "__main__":
    app.run_server(debug=True)
    




