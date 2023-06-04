import datetime
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow import keras

import numpy as np
import pandas as pd

OS = "LINUX" # "WINDOWS"


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

### LOAD AB TESTS RESULTS ###

if OS == "WINDOWS":
    AB_TESTS_RESULTS_FILEPATH = "..\\data\\ab_results.csv"
else:
    AB_TESTS_RESULTS_FILEPATH = "../data/ab_results.csv"

class ABResult(BaseModel):
    created_at: datetime.datetime
    user_id: int
    group: int
    model: int
    recomm_successful: bool


col_names = list(ABResult.__annotations__.keys())


def load_ab_tests():
    df_ab_tests = pd.DataFrame(columns=col_names)

    if os.path.isfile(AB_TESTS_RESULTS_FILEPATH):
        df_ab_tests = pd.read_csv(AB_TESTS_RESULTS_FILEPATH, index_col='index')

    return df_ab_tests

df_ab_tests = load_ab_tests()

### LOAD SESSION FILE ###

from src.recommender_playlist_provider.vae.VAEPlaylistProvider import VAEPlaylistProvider

if OS == "WINDOWS":
    DATA_DIR = "..\\data"
else:
    DATA_DIR = "../data"

with open(os.path.join(DATA_DIR, "tracks.json"), "rb") as f:
    tracks_features = pd.read_json(f)

with open(os.path.join(DATA_DIR, "users.json"), "rb") as f:
    users = pd.read_json(f)

sessions = pd.read_excel(os.path.join(DATA_DIR, "sessions.xlsx"))

with open(os.path.join(DATA_DIR, "available_users.npy"), "rb") as f:
    available_users = np.load(f, allow_pickle=True)


### LOADING ENCODER MODEL ###


if OS == "WINDOWS":
    ENCODER_DIR_FILES = "models\\encoder"
else:
    ENCODER_DIR_FILES = "models/encoder"### windows: "models\\encoder"

encoder = keras.models.load_model(os.path.join(ENCODER_DIR_FILES, "encoder_v4.h5"), compile=True)

with open(os.path.join(ENCODER_DIR_FILES, "embeddings_of_all_tracks.npy"), "rb") as f:
    embeddings_all_tracks = np.load(f, allow_pickle=True)

with open(os.path.join(ENCODER_DIR_FILES, "track_ids.npy"), "rb") as f:
    all_track_ids = np.load(f, allow_pickle=True)


class Input(BaseModel):
    user_id: int


# USER SESSIONS PROVIDER
from src.user_sessions_fetcher.common.UserSessionsProvider import UserSessionsProvider
from src.user_sessions_fetcher.common.constants import (EVENT_TYPE_PLAY, EVENT_TYPE_LIKE)

usp = UserSessionsProvider(sessions)


### API ENDPOINTS ###


@app.post("/models/{model_id}/predict")
def predict(model_id: int, input: Input):
    if input.user_id not in available_users:
        return {"error": f"data unavailable for user: {input.user_id}"}

    if model_id == 1:
        return model1_predict(input)
    elif model_id == 2:
        return model2_predict(input) #input.user_id
    else:
        return {"error": "model not found"}


def check_recommendations(user_id, recommended_tracks) -> bool:

    next_month_sessions = usp.get_user_sessions(user_id, event_types=[EVENT_TYPE_LIKE, EVENT_TYPE_PLAY],
                                                period_type='next')

    return bool(set(recommended_tracks) & set(next_month_sessions))


@app.post("/perform_ab_test")
def ab_test(input: Input):
    group = input.user_id % 2
    if group == 0:
        recommended_tracks = model1_predict(input)
    else:
        recommended_tracks = model2_predict(input)

    df_ab_tests = load_ab_tests()

    # recommended_tracks
    prediction = check_recommendations(input.user_id, recommended_tracks)

    df_ab_tests = df_ab_tests.append(
        {
            "created_at": pd.Timestamp.now(),
            "user_id": input.user_id,
            "group": group,
            "model": group + 1,
            "recomm_successful": prediction
        },
        ignore_index=True
    )

    # save file
    df_ab_tests.to_csv(AB_TESTS_RESULTS_FILEPATH, index_label='index')

    return {"Successful recommendation": prediction}


@app.delete("/ab_test/results")
def ab_test_clear_results():
    df_ab_tests = pd.DataFrame(columns=col_names)
    df_ab_tests.to_csv(AB_TESTS_RESULTS_FILEPATH, index_label='index')


@app.get("/ab_test/results")
def ab_test_results():
    return df_ab_tests.to_json()


def model1_predict(input: Input) -> List:
    session_track_ids_user_like = usp.get_user_sessions(input.user_id,
                                                        [EVENT_TYPE_LIKE], 'last')

    # number of recomm to produce based on number of tracks listened in the last month by user
    avg_n_of_tracks_in_user_sessions = usp.get_avg_n_of_tracks_in_user_sessions(input.user_id)

    vae_pp = VAEPlaylistProvider(model=encoder,
                                 embeddings_all_tracks=embeddings_all_tracks,
                                 ids_all_tracks=all_track_ids,
                                 all_tracks_features=tracks_features)

    recommendations = vae_pp.predict_recommendations(past_sessions_tracks_ids=session_track_ids_user_like,
                                                     n_of_tracks=avg_n_of_tracks_in_user_sessions)
    return recommendations


from src.recommender_playlist_provider.classifier.classifierPlaylistProvider import classifierPlaylistProvider
from src.track_preprocessor.ClassifierPreprocesor import classifierPreprocesor


if OS == "WINDOWS":
    SCALER_PATH = 'models\\classifier_track.scaler'
    MODEL_PATH = 'models\\classifier.model'
else:
    SCALER_PATH = 'models/classifier_track.scaler'
    MODEL_PATH = 'models/classifier.model'

pre = classifierPreprocesor(SCALER_PATH, OS=OS)
pre.prepare()

def model2_predict(input: Input) -> List:
    # number of recomm to produce based on number of tracks listened in the last month by user
    avg_n_of_tracks_in_user_sessions = usp.get_avg_n_of_tracks_in_user_sessions(input.user_id)
    p = classifierPlaylistProvider(MODEL_PATH, pre)
    x = p.predict_recommendations(n_of_tracks=avg_n_of_tracks_in_user_sessions, user_id=input.user_id)
    x = [z for z in x if z is not None]
    return x

if __name__ == "__main__":
    df_ab_tests = load_ab_tests()
    print(df_ab_tests)
    df_ab_tests = df_ab_tests.append(
        {
            "created_at": pd.Timestamp.now(),
            "user_id": 105,
            "group": 1,
            "model": 2,
            "recomm_successful": True
        },
        ignore_index=True
    )
    print(df_ab_tests)
    df_ab_tests.to_csv(AB_TESTS_RESULTS_FILEPATH, index_label='index')
    z = load_ab_tests()
    print(z)
