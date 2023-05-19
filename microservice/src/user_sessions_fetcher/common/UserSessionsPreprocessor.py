import pandas as pd
from datetime import datetime, timedelta
from typing import List

from src.common.Logger import Logger

from src.user_sessions_fetcher.common.constants import (
    TIMESTAMP_COLUMN_NAME,
    EVENT_TYPE_COLUMN_NAME
)

EVENT_TYPE_LIKE = 'like'


class UserSessionsPreprocessor:
    def __init__(self):
        Logger.info('[REPOSITORY] Creating User Sessions Preprocessor')

    def get_preprocessed_sessions(self, sessions: pd.DataFrame, event_types: List[str], period_type) -> pd.DataFrame:
        """
        Preprocesses data:
        1. Get data from last month
        2. Get event_type equal to like
        :param sessions: pd.DataFrame
        :return: pd.DataFrame
        """
        if period_type == 'last':
            filtered_sessions = self.get_sessions_from_last_month(sessions)
        else:
            filtered_sessions = self.get_sessions_from_next_month(sessions)
        filtered_sessions = self.get_sessions_event_type(filtered_sessions, event_types)
        return filtered_sessions.reset_index(drop=True)

    @staticmethod
    def get_sessions_from_last_month(sessions: pd.DataFrame) -> pd.DataFrame:
        # today = datetime.today()
        today = datetime.strptime("2021-11-01", "%Y-%m-%d")  # TODO: MOCK DATE for user 108 data
        last_month = today - timedelta(days=30)
        filtered_sessions = sessions[(sessions[TIMESTAMP_COLUMN_NAME] > last_month)
                                     & (sessions[TIMESTAMP_COLUMN_NAME] <= today)]
        return filtered_sessions

    @staticmethod
    def get_sessions_from_next_month(sessions: pd.DataFrame) -> pd.DataFrame:
        # today = datetime.today()
        today = datetime.strptime("2021-11-01", "%Y-%m-%d")  # TODO: MOCK DATE for user 108 data
        next_month = today + timedelta(days=30)
        filtered_sessions = sessions[(sessions[TIMESTAMP_COLUMN_NAME] <= next_month)
                                     & (sessions[TIMESTAMP_COLUMN_NAME] > today)]
        return filtered_sessions

    @staticmethod
    def get_sessions_event_type(sessions: pd.DataFrame, event_types: List[str]):
        return sessions[sessions[EVENT_TYPE_COLUMN_NAME].isin(event_types)]
