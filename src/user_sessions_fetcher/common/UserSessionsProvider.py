from typing import List

from common.Logger import Logger
from common.api.UserSessionsApi import UserSessionsApi
from src.common.constants import SESSION_ID_COLUMN_NAME, TIMESTAMP_COLUMN_NAME

from src.user_sessions_fetcher.common.constants import LOCATIONS_CONFIGURATIONS_API_URL_VAR, TRACK_ID_COLUMN_NAME
from src.user_sessions_fetcher.common.UserSessionsPreprocessor import UserSessionsPreprocessor
from src.user_sessions_fetcher.common.constants import (EVENT_TYPE_PLAY, EVENT_TYPE_LIKE)


class UserSessionsProvider:
    def __init__(self, SESSIONS_DIR="../data/test/"):
        Logger.info('[REPOSITORY] Creating User Sessions Provider')
        self.user_sessions_api = UserSessionsApi(LOCATIONS_CONFIGURATIONS_API_URL_VAR, SESSIONS_DIR)
        self.sessions_preprocessor = UserSessionsPreprocessor()

    def get_user_sessions(self, user_id: int,
                          event_types=[EVENT_TYPE_LIKE, EVENT_TYPE_PLAY],
                          period_type = 'last') -> List:
        """
        Fetches user sessions from API, preprocesses it and returns a list of tracks' IDs that user listened
        and liked in the last month.
        :param user_id: ID of user
        :return: List of track ids
        """
        try:
            user_sessions = self.user_sessions_api.get_user_sessions(user_id)
            preprocessed_sessions = self.sessions_preprocessor.get_preprocessed_sessions(user_sessions,
                                                                                         event_types,
                                                                                         period_type)
            track_ids = preprocessed_sessions[TRACK_ID_COLUMN_NAME].values
            return list(track_ids)
        except Exception as e:
            print(e)
            Logger.info(f"An exception occurred while fetching user sessions. {e}")

    def get_avg_n_of_tracks_in_user_sessions(self, user_id: int) -> int:
        try:
            user_sessions = self.user_sessions_api.get_user_sessions(user_id)
            last_month_sessions = self.sessions_preprocessor.get_sessions_from_last_month(user_sessions)
            tracks_per_session = last_month_sessions.groupby(SESSION_ID_COLUMN_NAME)[TIMESTAMP_COLUMN_NAME].count()
            return int(tracks_per_session.mean())
        except Exception as e:
            print(e)
            Logger.info(f"An exception occurred while fetching user sessions. {e}")
