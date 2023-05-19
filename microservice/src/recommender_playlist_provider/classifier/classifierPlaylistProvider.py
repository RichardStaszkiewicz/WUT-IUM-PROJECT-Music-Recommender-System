from recommender_playlist_provider.common.interfaces.PlaylistProviderBase import PlaylistProviderBase
from recommender_playlist_provider.classifier.classifierModel import Music_classifier, MusicDataset
import torch
import pandas as pd
import numpy as np
import joblib

class classifierPlaylistProvider(PlaylistProviderBase):
    # EMBEDDINGS_OF_ALL_TRACKS_FILENAME = "../../models/embeddings_of_all_tracks_3.npy"
    IDS_OF_ALL_TRACKS_FILENAME = "../../../models/track_ids_3.npy"
    FILE_WITH_ALL_TRACKS_FEATURES = "data/v2/tracks.json"
    FILE_WITH_ALL_SESSIONS = "data/v2/sessions.json"

    def __init__(self, model_path: str, scaler_path: str):
        super().__init__()
        self.model_path = model_path # '../../models/classifier.model'
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.sessions = None

    def predict_recommendations(self, past_sessions_tracks_ids, n_of_tracks, user):
        """
        Returns indices of n_of_tracks
        :param past_sessions_tracks_ids:
        :param n_of_tracks:
        :user: users.loc[x] ->  user_id                                                   102
                                name                                                      NaN
                                city                                                      NaN
                                street                                                    NaN
                                favourite_genres    [reggaeton, latin arena pop, modern rock]
                                premium_user                                            False
        :return:
        """
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        tracks = self.get_all_tracks()
        self.load_model_from_file(device)
        self.load_scaler_from_file()

        user = self.preprocess_user(user)
        tracks = self.preprocess_tracks(tracks)
        sessions = self.fetch_user_sessions(np.int64(user['user_id']))

        to_check = MusicDataset(user, tracks, sessions)

        return None

    def load_model_from_file(self, device):
        # Register the custom object
        self.model = Music_classifier(range(50))
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

    def load_scaler_from_file(self):
        self.scaler = joblib.load(self.scaler_path)

    def get_features_of_track_ids(self, tracks_ids):
        tracks_orig = pd.read_json(self.FILE_WITH_ALL_TRACKS_FEATURES)
        return tracks_orig[tracks_orig.id.isin(tracks_ids)]

    def get_all_tracks(self):
        return pd.read_json(self.FILE_WITH_ALL_TRACKS_FEATURES)

    def preprocess_tracks(self, tracks):
        self.track_ids = tracks.id
        VALID_COLUMN_NAMES = ['id', 'duration_ms', 'popularity', 'explicit', 'release_date','danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        tracks = tracks[VALID_COLUMN_NAMES]
        rd = tracks.release_date
        rd = pd.to_datetime(rd, errors='coerce')
        tracks['release_date'] = rd.dt.year.fillna(0).astype(int)
        track_keys = tracks['key']
        tracks = self.scaler.transform(tracks.drop(columns=['id', 'key']))
        tracks = pd.DataFrame(tracks, columns=['duration_ms', 'popularity', 'explicit', 'release_date','danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])
        tracks['id'] = self.track_ids
        tracks['key'] = track_keys
        return tracks

    def preprocess_user(self, user):
        user = user['user_id', 'favourite_genres', 'premium_user']
        user['premium_user'] = user['premium_user'] * 1
        return pd.DataFrame(user)

    def fetch_user_sessions(self, user_id, date=None):
        if self.sessions is None:
            self.sessions = pd.read_json(self.FILE_WITH_ALL_SESSIONS)
        if date is None:
            return self.sessions[self.sessions['user_id'] == user_id]
        else:
            return self.sessions # TBD: filtering by date break

if __name__ == "__main__":
    p = classifierPlaylistProvider('src/models/classifier.model', 'src/models/classifier_track.scaler')
    p.predict_recommendations(None, 5)
