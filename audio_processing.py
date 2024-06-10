import whisper
import subprocess
import torch
import pyannote.audio
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
import librosa
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import silhouette_score
from speechbrain.pretrained import SpeakerRecognition
import streamlit as st


def optimal_num_speakers(embeddings, min_speakers=1, max_speakers=5):
    """
    Determine the optimal number of speakers using silhouette score.

    Parameters:
    embeddings (np.array): Speaker embeddings.
    min_speakers (int): Minimum number of speakers to consider.
    max_speakers (int): Maximum number of speakers to consider.

    Returns:
    int: Optimal number of speakers.
    """
    best_num_clusters = min_speakers
    best_silhouette = -1  # Silhouette scores range from -1 to 1

    # Iterate over the range of specified cluster counts
    for n_clusters in range(min_speakers, max_speakers + 1):
        # Perform agglomerative clustering with n_clusters
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(embeddings)

        # Silhouette score wants a minimum of 2 clusters and at least 2 samples per cluster
        if len(set(labels)) > 1 and all(np.bincount(labels) > 1):
            silhouette_avg = silhouette_score(embeddings, labels)

            # Check if this silhouette score is the best
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_num_clusters = n_clusters

    return best_num_clusters

# Initialize the embedding model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

# embedding_model = SpeakerRecognition.from_hparams("pretrained_models/spkrec-ecapa-voxceleb")

def convert_to_wav(input_path):
    subprocess.call(['ffmpeg', '-i', input_path, 'audio.wav', '-y'])
    return 'audio.wav'

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def get_audio_metadata(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    frames = len(y)
    rate = sr
    return duration, frames, rate

@st.cache_data
def transcribe_audio(path, model_size='large'):
    model = whisper.load_model(model_size).to('cuda')
    result = model.transcribe(path)
    segments = result["segments"]
    return segments

@st.cache_data
def segment_embedding(path, duration, segment):
    audio = Audio()
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

def cluster_speakers(embeddings, num_speakers):
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    return labels


# def cluster_speakers(embeddings):
#     """
#     Apply clustering to the embeddings to identify speakers.

#     Parameters:
#     embeddings (np.array): Speaker embeddings.

#     Returns:
#     list: Cluster labels.
#     """
#     num_speakers = optimal_num_speakers(embeddings)
#     clustering_model = AgglomerativeClustering(num_speakers)
#     labels = clustering_model.fit_predict(embeddings)
#     return labels