from src.recommender_playlist_provider.common.interfaces.PlaylistProviderBase import PlaylistProviderBase
from src.recommender_playlist_provider.common.CallType import CallType
from src.recommender_playlist_provider.classifier.classifierModel import Music_classifier
import torch
import pandas as pd

class classifierPlaylistProvider(PlaylistProviderBase):
    # EMBEDDINGS_OF_ALL_TRACKS_FILENAME = "../../models/embeddings_of_all_tracks_3.npy"
    IDS_OF_ALL_TRACKS_FILENAME = "../../models/track_ids_3.npy"
    FILE_WITH_ALL_TRACKS_FEATURES = "../../../data/v2/tracks.json"

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path # '../../models/classifier.model'
        self.model = None

    def predict_recommendations(self, past_sessions_tracks_ids, n_of_tracks):
        """
        Returns indices of n_of_tracks
        :param past_sessions_tracks_ids:
        :param n_of_tracks:
        :return:
        """
        tracks = self.get_all_tracks()
        self.load_model_from_file()

        return None

    def load_model_from_file(self):
        # Register the custom object
        model = Music_classifier(range(50))
        model.load_state_dict(torch.load(self.model_path))

    def get_features_of_track_ids(self, tracks_ids):
        tracks_orig = pd.read_json(self.FILE_WITH_ALL_TRACKS_FEATURES)
        return tracks_orig[tracks_orig.id.isin(tracks_ids)]

    def get_all_tracks(self):
        return pd.read_json(self.FILE_WITH_ALL_TRACKS_FEATURES)

if __name__ == "__main__":
    p = classifierPlaylistProvider('../../models/classifier.model')
    x = pd.read_json("../../../data/v2/tracks.json")
    x.head()
    # p.predict_recommendations(None, 5)
