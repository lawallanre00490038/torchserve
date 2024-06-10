import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.cm as cm

def plot_2d_clusters(embeddings, segments, labels):
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, segment in enumerate(segments):
        speaker_id = labels[i] + 1
        x, y = embeddings_2d[i]
        plt.scatter(x, y, label=f'SPEAKER {speaker_id}')

    plt.title("Speaker Diarization Clusters (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

def plot_3d_clusters(embeddings, segments, labels):
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings)

    num_unique_speakers = len(np.unique(labels))
    colors = cm.tab20b(np.linspace(0, 1, num_unique_speakers))
    data = []

    for i, segment in enumerate(segments):
        speaker_id = labels[i] + 1
        x, y, z = embeddings_3d[i]
        color = colors[labels[i] % num_unique_speakers]
        trace = go.Scatter3d(x=[x], y=[y], z=[z], mode='markers',
                             marker=dict(size=5, color=color),
                             name=f'SPEAKER {speaker_id}')
        data.append(trace)

    layout = go.Layout(
        title="Speaker Diarization Clusters (3D Visualization)",
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
