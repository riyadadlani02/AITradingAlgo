import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1('Trading Portfolio Dashboard'),
    dcc.Textarea(
        id='portfolio-input',
        placeholder='Enter your portfolio details here...',
        style={'width': '100%', 'height': 100},
    ),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic', children='Enter a value and press submit')
])

# Define the callback to update the UI
@app.callback(
    Output('container-button-basic', 'children'),
    [Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('portfolio-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return f'Your portfolio: {value}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

