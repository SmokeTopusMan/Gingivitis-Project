import os
import io
import sys
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_google_drive():
    """Authenticate and return Google Drive service object."""
    creds = None
    
    # Check if token.json exists (stored credentials)
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Token refresh failed: {e}")
                creds = None
        
        if not creds:
            # You need to create credentials.json from Google Cloud Console
            if not os.path.exists('credentials.json'):
                print("ERROR: credentials.json not found!")
                print("\nTo set up Google Drive API access:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project or select existing one")
                print("3. Enable Google Drive API")
                print("4. Create credentials (OAuth 2.0 Client ID)")
                print("5. Download the JSON file and rename it to 'credentials.json'")
                print("6. Place credentials.json in the same directory as this script")
                sys.exit(1)
            
            # Manual authentication for headless servers using OOB flow
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            
            # Use out-of-band flow for headless servers
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            
            # Get authorization URL
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            print("\n" + "="*60)
            print("🔐 MANUAL AUTHENTICATION REQUIRED")
            print("="*60)
            print("\n1. Open this URL in your browser (on any device):")
            print(f"\n{auth_url}\n")
            print("2. Sign in and authorize the application")
            print("3. After authorization, Google will show you an authorization code")
            print("4. Copy that authorization code")
            print("5. Paste it below\n")
            
            code = input("Enter the authorization code: ").strip()
            
            if not code:
                print("❌ No authorization code provided. Exiting.")
                sys.exit(1)
            
            try:
                # Exchange code for credentials
                flow.fetch_token(code=code)
                creds = flow.credentials
                
                # Save the credentials for future runs
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
                
                print("✅ Authentication successful! Token saved for future use.")
                
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                print("Please check your authorization code and try again.")
                sys.exit(1)
    
    return build('drive', 'v3', credentials=creds)

def extract_folder_id(url):
    """Extract folder ID from Google Drive URL."""
    if '/folders/' in url:
        return url.split('/folders/')[1].split('?')[0]
    return url

def get_folder_contents(service, folder_id):
    """Get all files and subfolders in a Google Drive folder."""
    query = f"'{folder_id}' in parents and trashed=false"
    
    items = []
    page_token = None
    
    while True:
        try:
            results = service.files().list(
                q=query,
                pageSize=1000,  # Maximum allowed
                fields="nextPageToken, files(id, name, mimeType, parents)",
                pageToken=page_token
            ).execute()
            
            items.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            
            if not page_token:
                break
                
        except Exception as e:
            print(f"Error fetching folder contents: {e}")
            break
    
    return items

def download_file(service, file_id, file_name, download_path):
    """Download a single file from Google Drive."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        # Request file download
        request = service.files().get_media(fileId=file_id)
        
        # Download file
        with open(download_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Downloading {file_name}: {int(status.progress() * 100)}%", end='\r')
        
        print(f"✓ Downloaded: {file_name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {file_name}: {e}")
        return False

def download_folder_recursive(service, folder_id, local_path="./downloads", folder_name=""):
    """Recursively download all files from a Google Drive folder."""
    
    # Create local directory
    if folder_name:
        local_path = os.path.join(local_path, folder_name)
    
    os.makedirs(local_path, exist_ok=True)
    print(f"\n📁 Processing folder: {folder_name or 'Root'}")
    
    # Get folder contents
    items = get_folder_contents(service, folder_id)
    
    if not items:
        print(f"No files found in folder: {folder_name}")
        return
    
    files_count = 0
    folders_count = 0
    
    # Process all items
    for item in items:
        item_name = item['name']
        item_id = item['id']
        mime_type = item['mimeType']
        
        if mime_type == 'application/vnd.google-apps.folder':
            # It's a subfolder - recurse
            folders_count += 1
            print(f"📁 Found subfolder: {item_name}")
            download_folder_recursive(service, item_id, local_path, item_name)
        else:
            # It's a file - download it
            files_count += 1
            file_path = os.path.join(local_path, item_name)
            
            # Skip if file already exists
            if os.path.exists(file_path):
                print(f"⏭️  Skipping (already exists): {item_name}")
                continue
            
            download_file(service, item_id, item_name, file_path)
    
    print(f"\n📊 Folder '{folder_name or 'Root'}' summary:")
    print(f"   Files: {files_count}")
    print(f"   Subfolders: {folders_count}")

def main():
    """Main function."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Download all files from a Google Drive folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 gdrive_downloader.py "https://drive.google.com/drive/folders/1ABC123..."
  python3 gdrive_downloader.py "https://drive.google.com/drive/folders/1ABC123..." --output ./my_data
  python3 gdrive_downloader.py "https://drive.google.com/drive/folders/1ABC123..." -o /home/user/datasets
        """)
    
    parser.add_argument('folder_url', 
                       help='Google Drive folder URL')
    parser.add_argument('-o', '--output', 
                       default='./downloads',
                       help='Output directory for downloaded files (default: ./downloads)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate inputs
    if not args.folder_url:
        print("❌ Error: Please provide a Google Drive folder URL")
        parser.print_help()
        return
    
    if '/folders/' not in args.folder_url:
        print("❌ Error: Invalid Google Drive folder URL")
        print("   URL should look like: https://drive.google.com/drive/folders/FOLDER_ID")
        return
    
    # Extract folder ID
    folder_id = extract_folder_id(args.folder_url)
    print(f"📋 Folder URL: {args.folder_url}")
    print(f"📋 Folder ID: {folder_id}")
    print(f"📁 Output directory: {os.path.abspath(args.output)}")
    
    # Authenticate
    print("\n🔐 Authenticating with Google Drive...")
    try:
        service = authenticate_google_drive()
        print("✓ Authentication successful!")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return
    
    # Get folder info
    try:
        folder_info = service.files().get(fileId=folder_id).execute()
        folder_name = folder_info['name']
        print(f"📁 Downloading folder: {folder_name}")
    except Exception as e:
        print(f"✗ Could not access folder: {e}")
        print("Make sure the folder is publicly accessible or you have permission.")
        return
    
    # Start download
    print(f"\n🚀 Starting download...")
    download_folder_recursive(service, folder_id, args.output, folder_name)
    print(f"\n✅ Download complete! Files saved to: {os.path.abspath(os.path.join(args.output, folder_name))}")

if __name__ == "__main__":
    main()
