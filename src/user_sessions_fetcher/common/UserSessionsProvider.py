import os
from typing import List

from common.Logger import Logger
from common.api.UserSessionsApi import UserSessionsApi

from src.user_sessions_fetcher.common.constants import LOCATIONS_CONFIGURATIONS_API_URL_VAR, TRACK_ID_COLUMN_NAME
from src.user_sessions_fetcher.common.UserSessionsPreprocessor import UserSessionsPreprocessor

SESSIONS_DIR = "../data/test/"


class UserSessionsProvider:
    def __init__(self):
        Logger.info('[REPOSITORY] Creating User Sessions Provider')
        self.user_sessions_api = UserSessionsApi(LOCATIONS_CONFIGURATIONS_API_URL_VAR, SESSIONS_DIR)
        self.sessions_preprocessor = UserSessionsPreprocessor()

    def get_user_sessions(self, user_id: int) -> List:
        """
        Fetches user sessions from API, preprocesses it and returns a list of tracks' IDs that user listened
        and liked in the last month.
        :param user_id: ID of user
        :return: List of track ids
        """
        try:
            user_sessions = self.user_sessions_api.get_user_sessions(user_id)
            preprocessed_sessions = self.sessions_preprocessor.get_preprocessed_sessions(user_sessions)
            track_ids = list(preprocessed_sessions.reset_index(drop=True)[TRACK_ID_COLUMN_NAME].values)
            return track_ids
        except Exception as e:
            Logger.info(f"An exception occurred while fetching user sessions. {e}")
