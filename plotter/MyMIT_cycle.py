import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np

# File path and parameter settings
root = '../Dataset_plot/My data/MIT data'
folders = ['2017-05-12', '2017-06-30', '2018-04-12']
colors = ['#80A6E2', '#7BDFF2', '#FBDD85']
markers = ['circle', 'triangle-up', 'diamond']
legends = ['Batch 2017-05-12', 'Batch 2017-06-30', 'Batch 2018-04-12']
line_width = 1.0
window_size = 5

fig = go.Figure()

for i, folder in enumerate(folders):
    folder_path = os.path.join(root, folder)
    files = os.listdir(folder_path)

    for f in files:
        file_path = os.path.join(folder_path, f)
        data = pd.read_csv(file_path)

        # Extract and filter capacity
        capacity = data['capacity'].apply(lambda c: c if c <= 1.1 else np.nan)
        capacity_smoothed = capacity.ffill().rolling(window=window_size, min_periods=1).mean()

        # Use index as x-axis (cycle) just like your original code
        fig.add_trace(go.Scatter(
            x=list(range(len(capacity_smoothed))),
            y=capacity_smoothed,
            mode='lines+markers',
            marker=dict(symbol=markers[i], size=4, color=colors[i], line=dict(width=0.5)),
            line=dict(color=colors[i], width=line_width),
            name=legends[i],
            showlegend=False  # we'll add manual legend below
        ))

    # Add single legend entry per batch
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines+markers',
        marker=dict(symbol=markers[i], size=6, color=colors[i], line=dict(width=0.5)),
        line=dict(color=colors[i], width=line_width),
        name=legends[i]
    ))

# Set axis labels and layout (like in your original code)
fig.update_layout(
    xaxis_title='Cycle',
    yaxis_title='Capacity (Ah)',
    yaxis=dict(range=[0.8, 1.2]),
    width=600,
    height=300,
    margin=dict(l=50, r=20, t=20, b=50),
    font=dict(size=12),
    template='simple_white',
    legend=dict(x=0.7, y=0.95, font=dict(size=10), borderwidth=0)
)

fig.show()
