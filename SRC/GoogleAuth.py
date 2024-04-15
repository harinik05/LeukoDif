from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

# Class - Allows us to authenticate from the google drive API 
class GoogleDriveConnector:
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # Pass the credentials.json file as the input (Google Developers Console, OAuth)
    def __init__(self, credentials_file):
        self.credentials_file = credentials_file

    # Authenticate 
    def authenticate(self):
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json')

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)

            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        return creds

    # Access the google drive system after the token.json file is made
    def mount_drive(self):
        creds = self.authenticate()
        service = build('drive', 'v3', credentials=creds)

        root_folder_id = service.files().get(fileId='root').execute()['id']
        os.system(f"google-drive-ocamlfuse -headless -id {root_folder_id}")
        print("Mounted at /content/google_drive")
