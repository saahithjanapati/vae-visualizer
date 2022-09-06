from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px





app = Dash(__name__)
app.layout = html.Div([
    html.H6("Hello"),
    dcc.Graph(id="scatter-plot"),
    dcc.RangeSlider(
        id='range-slider',
        min=0, max=2.5, step=0.1,
        marks={0: '0', 2.5: '2.5'},
        value=[0.5, 2])
])


@app.callback(
    Output("scatter-plot", "figure"),
    Input("range-slider", "value"))

def update_bar_chart(slider_range):
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    return fig
    



if __name__ == "__main__":
    app.run_server(debug=True)
    




