import pydrive
import pickle
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from typing import Callable


class XlsxDriveLoader():
    """
    Read Excel files in Drive Folder
    """
    _loaded = None
    _folder_id = '1XoUes99yRgfm8KNT2Kf9ZXQBexos_2dm'

    def __new__(cls) -> dict:
        if ((databases := cls._loaded) is not None):
            print(
                'Database has already been loaded in memory and as a .pkl file in .data/')
            return databases

        databases = super().__new__(cls)
        cls._loaded = databases
        GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = './quickstart/client_secrets.json'
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        databases._load_databases(drive, cls._folder_id)

        return databases

    def _load_databases(self, drive, folder_id) -> None:

        data_container = drive.ListFile(
            {'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        load_databases: Callable[[pydrive.files.GoogleDriveFile], None] = (
            lambda x: x.FetchContent())
        gen = (map(load_databases, data_container))

        # Allocate data in memory
        while True:
            try:
                next(gen)
            except StopIteration:
                break

        # Structure databases
        databases = dict()
        for xlsx in data_container:
            if xlsx['title'].endswith('.xlsx'):
                xlsx_content = pd.ExcelFile(xlsx.content)

                xlsx_sheets = dict()
                for sheet_name in xlsx_content.sheet_names:
                    xlsx_sheets[sheet_name] = xlsx_content.parse(sheet_name)

                databases[xlsx['title']] = xlsx_sheets

        # Output
        self.content = databases
        with open('data/raw_databases.pkl', 'wb') as file:
            pickle.dump(databases, file)
