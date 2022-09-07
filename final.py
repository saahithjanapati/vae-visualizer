from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px


# TODO: make a thing to select two oints on the scatter plot
# TODO: make a gif maker class
# TODO: make slider worrk one-way
# TODO: integrate with model


SLIDER_MIN = 0
SLIDER_MAX = 9
SLIDER_INITIAL_VALUE = 5

app = Dash(__name__)
app.layout = html.Div([
    html.H6("Hello"),
    dcc.Graph(id="scatter-plot"),
    dcc.Slider(
        min=SLIDER_MIN,
        max=SLIDER_MAX,
        step=None,
        marks={i: str(i) for i in range(SLIDER_MIN, SLIDER_MAX)},
        value=SLIDER_INITIAL_VALUE, id="epoch-slider"),
    html.H6(f"Epoch number: {SLIDER_INITIAL_VALUE}", id="epoch-label")])



# this callback will eventually change the graph and the sliders, and the gifs
# pretty much everything on our dashboard
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
    




