import datetime
import os
import pickle
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

import numpy as np
import pandas as pd


app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

# USER SESSIONS PROVIDER
from src.user_sessions_fetcher.common.UserSessionsProvider import UserSessionsProvider
from src.user_sessions_fetcher.common.constants import (EVENT_TYPE_PLAY, EVENT_TYPE_LIKE)

usp = UserSessionsProvider("../data/test")



# LOAD session file #

from src.recommender_playlist_provider.vae.VAEPlaylistProvider import VAEPlaylistProvider

DATA_DIR = "..\\data"

with open(os.path.join(DATA_DIR, "tracks.json"), "rb") as f:
    tracks_features = pd.read_json(f)

### LOADING ENCODER MODEL ###

ENCODER_DIR_FILES = "models\\encoder"

encoder = keras.models.load_model(os.path.join(ENCODER_DIR_FILES, "encoder_v4.h5"), compile=True)

with open(os.path.join(ENCODER_DIR_FILES, "embeddings_of_all_tracks.npy"), "rb") as f:
    embeddings_all_tracks = np.load(f, allow_pickle=True)

with open(os.path.join(ENCODER_DIR_FILES, "track_ids.npy"), "rb") as f:
    all_track_ids = np.load(f, allow_pickle=True)


class Input(BaseModel):
    user_id: int


@app.post("/models/{model_id}/predict")
def predict(model_id: int, input: Input):
    if model_id == 1:
        return model1_predict(input)
    elif model_id == 2:
        return input.user_id
    else:
        return {"error": "model not found"}


def model1_predict(input: Input) -> List:
    session_track_ids_user_108_like = usp.get_user_sessions(input.user_id,
                                                            [EVENT_TYPE_LIKE], 'last')

    # number of recomm to produce based on number of tracks listened in the last month by user
    avg_n_of_tracks_in_user_sessions = usp.get_avg_n_of_tracks_in_user_sessions(input.user_id)

    vae_pp = VAEPlaylistProvider(model=encoder,
                                 embeddings_all_tracks=embeddings_all_tracks,
                                 ids_all_tracks=all_track_ids,
                                 all_tracks_features=tracks_features)

    recommendations = vae_pp.predict_recommendations(past_sessions_tracks_ids=session_track_ids_user_108_like,
                                                     n_of_tracks=avg_n_of_tracks_in_user_sessions)
    return recommendations