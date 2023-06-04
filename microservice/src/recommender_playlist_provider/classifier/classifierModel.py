import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

class Music_classifier(nn.Module):
    def __init__(self, genres):
        self.genres = genres
        #    |
        # 50 | Genres | 10    }
        #    |                } USER [11]
        # czy_premium | 1     }

        #    |
        # 16 | key    | 3     }
        #    |                } TRACK [16]
        # numeric     | 12    }
        # binary      | 1     }

        track_key_code = 3
        user_genre_code = 10
        super(Music_classifier, self).__init__()
        ##### USER PREP ######
        self.emb_user_genre = nn.Linear(len(genres), user_genre_code)
        self.emb_user_genre_act = nn.LeakyReLU()

        ##### TRACK PREP #####
        self.emb_track_key = nn.Linear(16, track_key_code)
        self.emb_track_key_act = nn.LeakyReLU()

        #### MAIN CLASSIFIER ####
        user_params = user_genre_code + 1
        track_params = track_key_code + 12 + 1

        self.layers = nn.Sequential(
            nn.Linear(user_params + track_params, 128),
            nn.LeakyReLU(),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),

            nn.Sigmoid()
        )
    def forward(self, x):

        user, track = x

        comp_emb_user_genre = self.emb_user_genre(user[0])
        comp_emb_user_genre = self.emb_user_genre_act(comp_emb_user_genre)

        comp_emb_track_key = self.emb_track_key(track[0])
        comp_emb_track_key = self.emb_track_key_act(comp_emb_track_key)

        c = torch.cat([comp_emb_user_genre, user[1], comp_emb_track_key, track[1]], 1)

        return self.layers(c)

class MusicDataset(data.Dataset):
    def __init__(self, users, tracks, sessions):
        self.users = users
        self.lusers = len(users)
        self.tracks = tracks
        self.ltracks = len(tracks)
        self.sessions = sessions
        self.genres = range(50)
        self.z = max(self.lusers, self.ltracks)

    def __len__(self):
        return self.ltracks * self.lusers

    def __getitem__(self, idx):
        u = self.users.loc[idx // self.z]
        t = self.tracks.loc[idx % self.z]
        select = self.sessions[u['user_id'] == self.sessions['user_id']]
        select = select[t['id'] == select['track_id']]
        play = bool(sum(select['event_type'] == 'play'))
        like = bool(sum(select['event_type'] == 'like'))
        ug = u['favourite_genres']
        ug = torch.Tensor(np.array([i in ug for i in self.genres]) * 1)
        ur = torch.Tensor([u['premium_user'] * 1])
        tk = torch.Tensor([int(i == t['key']) for i in range(16)])
        tr = torch.Tensor(t.drop(labels=['id', 'key']).to_numpy().astype(np.float64))

        return [[ug, ur], [tk, tr]], [play, like]