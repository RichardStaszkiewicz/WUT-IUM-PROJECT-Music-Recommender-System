import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from src.recommender_playlist_provider.common.interfaces.PlaylistProviderBase import PlaylistProviderBase
from src.track_preprocessor.VAEPreprocessor import VAEPreprocessor
from src.recommender_playlist_provider.common.CallType import CallType


class VAEPlaylistProvider(PlaylistProviderBase):
    def __init__(self, model, embeddings_all_tracks,
                 ids_all_tracks, all_tracks_features):
        super().__init__()
        self.vae_preprocessor = VAEPreprocessor()
        self.model = model
        self.embeddings_all_tracks = embeddings_all_tracks
        self.ids_all_tracks = ids_all_tracks
        self.tracks = all_tracks_features

    def get_embeddings_of_all_tracks(self):
        return self.embeddings_all_tracks

    def get_features_of_track_ids(self, tracks_ids):
        return self.tracks[self.tracks.id.isin(tracks_ids)]

    def get_indices_of_all_tracks(self):
        return self.ids_all_tracks

    def predict_recommendations(self, past_sessions_tracks_ids, n_of_tracks):
        """
        Returns indices of n_of_tracks
        :param past_sessions_tracks_ids:
        :param n_of_tracks:
        :return:
        """
        past_sessions_tracks = self.get_features_of_track_ids(past_sessions_tracks_ids)

        embedded_tracks = self._embed_tracks_to_latent_space(past_sessions_tracks)
        embedded_all_tracks = self.get_embeddings_of_all_tracks()
        tracks_ids = self.get_indices_of_all_tracks()

        N_RECOMM_PER_TRACK = 1
        tracks = []
        track_distances = []
        for embedded_track_idx in range(embedded_tracks.shape[0]):
            embedded_track = embedded_tracks[embedded_track_idx, ]
            recomm_tracks_ids, distances = self.find_n_closest_tracks(embedded_track, embedded_all_tracks, n=N_RECOMM_PER_TRACK)
            tracks.extend(recomm_tracks_ids)
            track_distances.extend(distances)
        track_ids_and_distances = np.c_[np.array(tracks), np.array(track_distances)]
        sorted_indices = np.argsort(track_ids_and_distances[:, 1])

        ids_of_tracks_to_recomm = track_ids_and_distances[sorted_indices][:n_of_tracks, 0].astype(int)
        return list(tracks_ids[ids_of_tracks_to_recomm])

    def _embed_tracks_to_latent_space(self, tracks):
        preprocessed_tracks = self.vae_preprocessor.preprocess_tracks(tracks_data=tracks,
                                                                      call_type=CallType.INFERENCE)
        latent_space = self.model.predict(preprocessed_tracks)
        return latent_space

    def find_n_closest_tracks(self, input_track_embedding, all_tracks_embeddings, n):
        # Calculate the Euclidean distances between the input track and all tracks
        distances = np.linalg.norm(all_tracks_embeddings - input_track_embedding, axis=1)
        closest_indices = np.argsort(distances)[:n]
        closest_distances = distances[closest_indices]
        return closest_indices, closest_distances
