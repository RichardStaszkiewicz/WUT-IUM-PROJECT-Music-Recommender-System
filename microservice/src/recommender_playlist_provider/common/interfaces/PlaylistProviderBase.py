from abc import ABC, abstractmethod
from typing import List

from src.user_sessions_fetcher.common.UserSessionsProvider import UserSessionsProvider


class PlaylistProviderBase(ABC):

    def __init__(self):
        self.user_session_provider = UserSessionsProvider()

    def get_last_user_session(self, user_id: int):
        return self.user_session_provider.get_user_sessions(user_id)

    def get_average_number_of_tracks_in_user_last_month_sessions(self, user_id: int):
        return self.user_session_provider.get_avg_n_of_tracks_in_user_sessions(user_id)

    def get_recommended_playlist_for_user(self, user_id: int) -> List:
        """
        Provides a track recommendations for a given user in a form of playlist.
        :param user_id: Id of user.
        :return: List of recommended track ids
        """
        preprocessed_sessions = self.get_last_user_session(user_id)
        n_of_tracks_to_recommend = self.get_average_number_of_tracks_in_user_last_month_sessions(user_id)
        return self.predict_recommendations(preprocessed_sessions, n_of_tracks_to_recommend)

    @abstractmethod
    def predict_recommendations(self, past_sessions, n_of_tracks):
        pass
