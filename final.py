from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px


# TODO: make a thing to select two points on the scatter plot
# TODO: make a gif maker class
# TODO: make slider worrk one-way
# TODO: integrate with model
# TODO: plain latent space explorer (just adjust vector values and see what happens ???)


SLIDER_MIN = 0
SLIDER_MAX = 9
SLIDER_INITIAL_VALUE = 9
DATA_POINT_1 = None
DATA_POINT_2 = None
STATIC_GRAPH = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

RADIO_BUTTONS = ["Data Point 1", "Data Point 2"]
INITAL_RADIO_BUTTON = RADIO_BUTTONS[0]


app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="scatter-plot", figure=STATIC_GRAPH),
    dcc.Slider(
        min=SLIDER_MIN,
        max=SLIDER_MAX,
        step=None,
        marks={i: str(i) for i in range(SLIDER_MIN, SLIDER_MAX)},
        value=SLIDER_INITIAL_VALUE, id="epoch-slider"),
    html.H6(f"Epoch number: {SLIDER_INITIAL_VALUE}", id="epoch-label"),
    dcc.RadioItems(RADIO_BUTTONS, INITAL_RADIO_BUTTON, id="radio_button"),
    html.H6(f"Data Point Being Edited: {INITAL_RADIO_BUTTON}", id="radio_info")])





# this callback will eventually change the graph and the sliders, and the gifs
# pretty much everything on our dashboard
@app.callback(
    Output("radio_info", "children"),
    Input("radio_button", "value"))
def update_radio_selection(option):
    return f"Data Point Being Edited: {option}"


@app.callback(
    Output("epoch-label", "children"),
    Input("epoch-slider", "value"))
def update_epoch_number(epoch_number):
    return f"Epoch number: {epoch_number}" 



# update graph according to epoch slider value
# @app.callback(
#     Output("scatter-plot", "figure"),
#     Input("epoch-slider", "value"))
# def update_bar_chart(slider_range):
#     fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
#     return fig






if __name__ == "__main__":
    app.run_server(debug=True)
    




