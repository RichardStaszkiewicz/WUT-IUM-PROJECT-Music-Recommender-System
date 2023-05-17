import pandas as pd
from sklearn.preprocessing import StandardScaler
from enum import Enum
import json
from src.common.constants import (RELEASE_DATA_COLUMN_NAME,
                                  KEY_COLUMN_NAME)
from src.recommender_playlist_provider.common.CallType import CallType

VALID_COLUMN_NAMES = ['duration_ms', 'popularity', 'explicit', 'release_date', 'danceability', 'energy', 'key',
                      'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
SCALES_PATH = "../../models/scales/standarization_scales.json"
KEY_CATEGORIES_PATH = "../../models/scales/key_categories.json"


class VAEPreprocessor:
    def preprocess_tracks(self, tracks_data: pd.DataFrame, call_type: CallType = CallType.INFERENCE):
        if call_type.value == CallType.INFERENCE.value:
            sliced_data = tracks_data[VALID_COLUMN_NAMES]
            preprocessed_tracks = self.simplify_date_column(sliced_data)

            non_key_data = preprocessed_tracks.drop(columns=[KEY_COLUMN_NAME], inplace=False)
            standardized_columns = self.standardize_columns(non_key_data, call_type)

            encoded_keys = self.one_hot_encode_data(preprocessed_tracks[KEY_COLUMN_NAME], call_type)
            encoded_keys = encoded_keys.sort_index(axis=1)
        elif call_type.value == CallType.TRAINING.value:
            sliced_data = tracks_data[VALID_COLUMN_NAMES]
            preprocessed_tracks = self.simplify_date_column(sliced_data)

            non_key_data = preprocessed_tracks.drop(columns=[KEY_COLUMN_NAME], inplace=False)
            standardized_columns = self.standardize_columns(non_key_data, call_type)

            encoded_keys = self.one_hot_encode_data(preprocessed_tracks[KEY_COLUMN_NAME], call_type)

        return pd.concat([standardized_columns.reset_index(drop=True),
                          encoded_keys.reset_index(drop=True)], axis=1)

    @staticmethod
    def simplify_date_column(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if isinstance(data, pd.DataFrame):
            data = data.copy()
            datetime_dates = pd.to_datetime(data[RELEASE_DATA_COLUMN_NAME], errors='coerce')
            dates_int = datetime_dates.dt.year
            data.loc[:, RELEASE_DATA_COLUMN_NAME] = dates_int
        elif isinstance(data, pd.Series):
            datetime_dates = pd.to_datetime(data, errors='coerce')
            dates_int = datetime_dates.dt.year
            data = dates_int
        else:
            raise ValueError("Unsupported data type. Expected pd.DataFrame or pd.Series.")
        return data

    def standardize_columns(self, df, call_type) -> pd.DataFrame:
        if call_type.value == CallType.TRAINING.value:
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(df)
            self._save_standardization_parameters(scaler.mean_, scaler.scale_, SCALES_PATH)
        elif call_type.value == CallType.INFERENCE.value:
            loaded_mean, loaded_std = self._load_standardization_parameters(SCALES_PATH)
            scaler = StandardScaler()
            scaler.mean_ = loaded_mean
            scaler.scale_ = loaded_std
            standardized_data = scaler.transform(df)
        return pd.DataFrame(standardized_data, columns=df.columns)

    @staticmethod
    def _save_standardization_parameters(mean: list, std: list, filepath: str):
        params = {
            'mean': list(mean),
            'std': list(std)
        }
        with open(filepath, 'w+') as file:
            json.dump(params, file)

    @staticmethod
    def _load_standardization_parameters(filepath: str):
        with open(filepath, 'r') as file:
            params = json.load(file)
        return params['mean'], params['std']

    def one_hot_encode_data(self, df: pd.Series, call_type) -> pd.Series:
        if call_type.value == CallType.TRAINING.value:
            # Get unique categories from the column
            unique_categories = [int(x) for x in list(df.unique())]
            # Save the unique categories to a file
            self.save_unique_categories(unique_categories, KEY_CATEGORIES_PATH)
            return pd.get_dummies(df)
        elif call_type.value == CallType.INFERENCE.value:
            categories = self.load_unique_categories(KEY_CATEGORIES_PATH)
            return pd.get_dummies(df).reindex(columns=categories, fill_value=0)

    @staticmethod
    def save_unique_categories(categories: list, filepath: str):
        with open(filepath, 'w') as file:
            json.dump(categories, file)

    @staticmethod
    def load_unique_categories(filepath: str) -> list:
        with open(filepath, 'r') as file:
            categories = json.load(file)
        return categories
