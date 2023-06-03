import pandas as pd
import numpy as np
import joblib

class classifierPreprocesor(object):

    def __init__(self, scaler_path, genres=None, OS=None) -> None:
        self.sessions = None
        self.tracks = None
        self.users = None
        self.scaler = None
        self.scaler_path = scaler_path
        self.genres = genres
        if OS == "WINDOWS": #DEFAULT: LINUX convention
            self.FILE_WITH_ALL_TRACKS_FEATURES = "..\\data\\batch1\\tracks.json"
            self.FILE_WITH_ALL_SESSIONS = "..\\data\\batch1\\sessions_100.json" #"data/v2/sessions.json" "../data/v2/sessions.json"
            self.FILE_WITH_ALL_USERS = "..\\data\\v2\\users.json"
        else:
            self.FILE_WITH_ALL_TRACKS_FEATURES = "../data/batch1/tracks.json"
            self.FILE_WITH_ALL_SESSIONS = "../data/batch1/sessions_100.json" # "../data/batch1/sessions_100.json" #"data/v2/sessions.json"
            self.FILE_WITH_ALL_USERS = "../data/v2/users.json"


    def prepare(self):
        self.scaler = self.load_scaler_from_file(self.scaler_path)
        self.tracks = self.load_all_tracks()
        self.tracks = self.preprocess_tracks(self.tracks)
        self.sessions = self.load_all_sessions()
        self.sessions = self.preprocess_sessions(self.sessions)
        if self.genres is None:
            self.genres = self.load_genres()

    def get_sessions(self):
        return self.sessions

    def get_tracks(self):
        return self.tracks

    def get_genres(self):
        return self.genres

    def get_users(self):
        return self.users

    def load_all_tracks(self):
        return pd.read_json(self.FILE_WITH_ALL_TRACKS_FEATURES)

    def load_all_users(self):
        return pd.read_json(self.FILE_WITH_ALL_USERS)

    def get_features_of_track_ids(self, tracks_ids):
        return self.tracks[self.tracks.id.isin(tracks_ids)]

    def load_all_sessions(self):
        return pd.read_json(self.FILE_WITH_ALL_SESSIONS)

    def load_genres(self):
        if self.users is None:
            self.users = pd.read_json(self.FILE_WITH_ALL_USERS)
        return np.unique(np.concatenate(self.users['favourite_genres'].to_numpy()))

    def get_user_sessions(self, user_id, date=None):
        if date is None:
            return self.sessions[self.sessions['user_id'] == user_id]
        else:
            return self.sessions # TBD: filtering by date break

    def get_dummie_sessions(self):
        return self.sessions.head()

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

    def preprocess_sessions(self, sessions):
        to_drop = sessions['event_type'] == 'skip'
        to_drop[len(to_drop)] = False
        to_drop = [ (to_drop[i+1] or to_drop[i]) for i in range(len(to_drop) - 1) ]
        to_drop = np.array(to_drop)
        to_drop += sessions['event_type'] == 'advertisment'
        to_drop = ~to_drop
        sessions = sessions[to_drop]
        return sessions

    def preprocess_user(self, user_id):
        self.users = self.load_all_users()
        user = self.users[self.users['user_id'] == user_id]
        user = user[['user_id', 'favourite_genres', 'premium_user']]
        user['premium_user'] = user['premium_user'] * 1
        if type(user) != pd.DataFrame:
            user = pd.DataFrame(user).transpose()
        return user.reset_index(drop=True)

    def load_scaler_from_file(self, scaler_path):
        return joblib.load(scaler_path)

if __name__ == "__main__":
    pre = classifierPreprocesor('src/models/classifier_track.scaler')
    pre.prepare()