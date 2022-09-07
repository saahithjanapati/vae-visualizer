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
INITIAL_RADIO_SELECTION = RADIO_BUTTONS[0]
RADIO_SELECTION = RADIO_BUTTONS[0]


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
    dcc.RadioItems(RADIO_BUTTONS, INITIAL_RADIO_SELECTION, id="radio_button"),
    html.H6(f"Data Point Being Edited: {INITIAL_RADIO_SELECTION}", id="radio_info"),
    
    
    html.H6( f"{RADIO_BUTTONS[0]} Value: {DATA_POINT_1}", id="point1"),
    html.H6(f"{RADIO_BUTTONS[1]} Value: {DATA_POINT_2}", id="point2")])


# this callback will eventually change the graph and the sliders, and the gifs
# pretty much everything on our dashboard
@app.callback(
    Output("radio_info", "children"),
    Input("radio_button", "value"))
def update_radio_selection(option):
    global RADIO_SELECTION
    RADIO_SELECTION = option
    print(f"RADIO_SELECTION: {RADIO_SELECTION}")
    return f"Data Point Being Edited: {RADIO_SELECTION}"


@app.callback(
    Output("epoch-label", "children"),
    Input("epoch-slider", "value"))
def update_epoch_number(epoch_number):
    return f"Epoch number: {epoch_number}" 


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
    




