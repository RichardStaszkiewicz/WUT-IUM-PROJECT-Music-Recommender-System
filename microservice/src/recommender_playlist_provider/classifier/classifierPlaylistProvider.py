from src.recommender_playlist_provider.common.CallType import CallType
from src.recommender_playlist_provider.classifier.classifierModel import Music_classifier, MusicDataset
from src.track_preprocessor.ClassifierPreprocesor import classifierPreprocesor

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd


class classifierPlaylistProvider:

    def __init__(self, model_path: str, preprocesor: classifierPreprocesor):
        super().__init__()
        self.model_path = model_path # '../../models/classifier.model'
        self.preprocesor = preprocesor
        self.model = None

    def predict_recommendations(self, n_of_tracks, user_id):
        """
        Returns indices of n_of_tracks
        :param past_sessions_tracks_ids:
        :param n_of_tracks:
        :user_id:
        :return:
        """
        device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.load_model_from_file(device)

        user = self.preprocesor.preprocess_user(user_id)
        sessions = self.preprocesor.get_user_sessions(np.int64(user['user_id'][0]))
        tracks = self.preprocesor.get_tracks()
        to_check = MusicDataset(user, tracks, sessions)
        to_check.genres = self.preprocesor.get_genres()
        loader = data.DataLoader(to_check, batch_size=64, shuffle=False, drop_last=False)

        answer = pd.DataFrame()
        answer['track_id'] = tracks['id']
        answer['play'] = 0
        answer['like'] = 0
        pos = 0
        self.model.eval()
        for _ in range(5):#range(len(user) * len(tracks)):
            x, _ = next(iter(loader))
            x = [ [ x[0][0].to(device), x[0][1].to(device) ], [ x[1][0].to(device), x[1][1].to(device) ] ]
            pred = self.model(x)
            for ans in pred:
                answer.loc[pos, 'play'] = np.float64(ans[0])
                answer.loc[pos, 'like'] = np.float64(ans[1])
                pos += 1
        answer['weights'] = answer['play'] + answer['like'] + np.random.rand(len(answer))/10

        return answer['track_id'].sample(n_of_tracks, weights=answer['weights']).to_numpy()

    def load_model_from_file(self, device):
        # Register the custom object
        self.model = Music_classifier(self.preprocesor.get_genres())
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

if __name__ == "__main__":
    pre = classifierPreprocesor('models/classifier_track.scaler')
    pre.prepare()
    p = classifierPlaylistProvider('models/classifier.model', pre)
    print(p.predict_recommendations(5, user_id=101))#pd.read_json("data/v2/users.json").loc[0]))
