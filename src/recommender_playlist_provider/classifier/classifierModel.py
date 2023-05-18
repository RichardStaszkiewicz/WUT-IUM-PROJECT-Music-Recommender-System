import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

GENRES = range(50)

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

        # TRACK [16] |        }
        # TRACK [16] |   | 10 } FINGERPRINT [10]
        # TRACK [16] |        }
        track_key_code = 3
        user_genre_code = 10
        fingerprint_params = 15
        super(Music_classifier, self).__init__()
        ##### USER PREP ######
        self.emb_user_genre = nn.Linear(len(genres), user_genre_code)
        self.emb_user_genre_act = nn.LeakyReLU()

        ##### TRACK PREP #####
        self.emb_track_key = nn.Linear(16, track_key_code)
        self.emb_track_key_act = nn.LeakyReLU()

        ##### FINGERPRINT PREP #####
        # self.emb_fingerprint = nn.Linear(3 * (track_key_code + 12 + 1), fingerprint_params)
        # self.emb_fingerprint_act = nn.LeakyReLU()
        # self.emb_fingerprint_track1 = nn.Linear(16, track_key_code)
        # self.emb_fingerprint_track1_act = nn.LeakyReLU()
        # self.emb_fingerprint_track2 = nn.Linear(16, track_key_code)
        # self.emb_fingerprint_track2_act = nn.LeakyReLU()
        # self.emb_fingerprint_track3 = nn.Linear(16, track_key_code)
        # self.emb_fingerprint_track3_act = nn.LeakyReLU()

        #### MAIN CLASSIFIER ####
        user_params = user_genre_code + 1
        track_params = track_key_code + 12 + 1

        self.layers = nn.Sequential(
            # nn.Linear(user_params + track_params + fingerprint_params, 256),
            nn.Linear(user_params + track_params, 128),
            nn.LeakyReLU(),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            #nn.Linear(32, 3),
            nn.Linear(32, 2),

            nn.Sigmoid()
        )
    #def forward(self, user, track, finger):
    def forward(self, x):
        # user -> [Tensor(50, bs), Tensor(1, bs)]
        # track -> [Tensor(16, bs), Tensor(13, bs)]
        # finger -> [ [Tensor(16, bs), Tensor(13, bs)], [Tensor(16, bs), Tensor(13, bs)], [Tensor(16, bs), Tensor(13, bs)] ]
        user, track = x

        comp_emb_user_genre = self.emb_user_genre(user[0])
        comp_emb_user_genre = self.emb_user_genre_act(comp_emb_user_genre)

        comp_emb_track_key = self.emb_track_key(track[0])
        comp_emb_track_key = self.emb_track_key_act(comp_emb_track_key)

        # comp_emb_fingerprint_track1 = self.emb_fingerprint_track1(finger[0][0])
        # comp_emb_fingerprint_track1 = self.emb_fingerprint_track1_act(comp_emb_fingerprint_track1)
        # comp_emb_fingerprint_track2 = self.emb_fingerprint_track2(finger[1][0])
        # comp_emb_fingerprint_track2 = self.emb_fingerprint_track2_act(comp_emb_fingerprint_track2)
        # comp_emb_fingerprint_track3 = self.emb_fingerprint_track3(finger[2][0])
        # comp_emb_fingerprint_track3 = self.emb_fingerprint_track3_act(comp_emb_fingerprint_track3)
        # emb_finger_x = torch.cat([comp_emb_fingerprint_track1, finger[0][1],
        #                           comp_emb_fingerprint_track2, finger[1][1],
        #                           comp_emb_fingerprint_track3, finger[2][1]], dim=1)
        # comp_emb_fingerprint = self.emb_fingerprint(emb_finger_x)
        # comp_emb_fingerprint = self.emb_fingerprint_act(comp_emb_fingerprint)

        # x = torch.cat([comp_emb_user_genre, user[1], comp_emb_track_key, track[1], comp_emb_fingerprint])
        # print(comp_emb_user_genre.shape, user[1].shape, comp_emb_track_key.shape, track[1].shape)
        c = torch.cat([comp_emb_user_genre, user[1], comp_emb_track_key, track[1]], 1)

        return self.layers(c)

class MusicDataset(data.Dataset):
    def __init__(self, users, tracks, sessions):
        self.users = users
        self.lusers = len(users)
        self.tracks = tracks
        self.ltracks = len(tracks)
        self.sessions = sessions
        self.z = min(self.lusers, self.ltracks)

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
        ug = torch.Tensor(np.array([i in ug for i in GENRES]) * 1)
        ur = torch.Tensor([u['premium_user'] * 1])
        tk = torch.Tensor([int(i == t['key']) for i in range(16)])
        tr = torch.Tensor(t.drop(labels=['id', 'key']).to_numpy().astype(np.float64))

        return [[ug, ur], [tk, tr]], [play, like]