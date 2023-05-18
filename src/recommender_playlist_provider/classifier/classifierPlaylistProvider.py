from src.recommender_playlist_provider.common.interfaces.PlaylistProviderBase import PlaylistProviderBase
from src.recommender_playlist_provider.common.CallType import CallType
from src.recommender_playlist_provider.classifier.classifierModel import Music_classifier, MusicDataset
from src.track_preprocessor.ClassifierPreprocesor import classifierPreprocesor
import torch
import pandas as pd
import numpy as np
import joblib

class classifierPlaylistProvider(PlaylistProviderBase):
    # EMBEDDINGS_OF_ALL_TRACKS_FILENAME = "../../models/embeddings_of_all_tracks_3.npy"
    IDS_OF_ALL_TRACKS_FILENAME = "../../models/track_ids_3.npy"

    def __init__(self, model_path: str, preprocesor: classifierPreprocesor):
        super().__init__()
        self.model_path = model_path # '../../models/classifier.model'
        self.preprocesor = preprocesor
        self.model = None

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
        self.load_model_from_file(device)

        user = self.preprocesor.preprocess_user(user)
        sessions = self.preprocesor.get_user_sessions(np.int64(user['user_id'][0]))

        to_check = MusicDataset(user, self.preprocesor.get_tracks(), sessions)

        return None

    def load_model_from_file(self, device):
        # Register the custom object
        self.model = Music_classifier(range(50))
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

if __name__ == "__main__":
    pre = classifierPreprocesor('src/models/classifier_track.scaler')
    pre.prepare()
    print(pre.get_user_sessions(101))

    # p = classifierPlaylistProvider('src/models/classifier.model', 'src/models/classifier_track.scaler')
    # p.predict_recommendations(None, 5, user=pd.read_json("data/v2/users.json").loc[0])
