import pandas as pd
import os

from microservice.src.common.api.interfaces.ApiClientBase import ApiClientBase
from microservice.src.common.constants import TIMESTAMP_COLUMN_NAME

"""
Mock class of UserSessionsApi.
Fetches sessions data from user of id 108.
"""


class UserSessionsApi(ApiClientBase):
    def __init__(self, api_url, sessions_dir):
        super().__init__(api_url, None)
        self.sessions_dir = sessions_dir

    def get_session_files(self):
        session_files = []
        for file in os.listdir(self.sessions_dir):
            if file.startswith("sessions_user_108"):
                session_files.append(os.path.join(self.sessions_dir, file))
        return session_files

    def get_user_sessions(self, user_id):
        sessions = pd.DataFrame([])
        for session_file in self.get_session_files():
            sessions = pd.concat([pd.read_excel(session_file, parse_dates=[TIMESTAMP_COLUMN_NAME]), sessions])
        return sessions

    # def get_sessions(self):
    #     response = requests.get(self.api_url,
    #                             headers=self.headers)
    #     return response.json()


