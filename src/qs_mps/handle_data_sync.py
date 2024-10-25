import subprocess

# Define server details
servers = [
    {"hostname": "158.227.6.203", "username": "fradm", "local_path": "Desktop/projects/1_Z2/results"},
    {"hostname": "158.227.46.38", "username": "fradm", "local_path": "Desktop/projects/1_Z2/results"},
    {"hostname": "158.227.47.136", "username": "fradm", "local_path": "Desktop/projects/1_Z2/results"}
]

# Define the local directory to sync to
local_dir = "C:/Users/HP/Desktop/projects/1_Z2/results"

# Function to generate a filename or search pattern based on parameters
def generate_filename(params):
    filename = "_".join(f"{key}_{value}" for key, value in params.items())
    return filename

# Use rsync to fetch result from any server where it exists
def fetch_result(params):
    filename_pattern = generate_filename(params)
    for server in servers:
        remote_path = f"{server['username']}@{server['hostname']}:{server['local_path']}/{filename_pattern}*"
        
        # Rsync command to sync matching files to the local directory
        rsync_command = ["rsync", "-avz", "--ignore-missing-args", remote_path, local_dir]
        
        try:
            result = subprocess.run(rsync_command, capture_output=True, text=True, check=True)
            if result.returncode == 0 and "skipping non-regular file" not in result.stdout:
                print(f"Result synced from {server['hostname']}")
                return f"{local_dir}/{filename_pattern}"
        except subprocess.CalledProcessError as e:
            print(f"Error syncing from {server['hostname']}: {e}")
    
    print("Result not found on any server. Computation may still be required.")
    return None


# Local path of the Google Drive folder to copy
google_drive_path = "G:/My Drive/projects/1_Z2/results/"

# Function to copy Google Drive folder to each server using scp
def copy_google_drive_scp():
    for server in servers:
        remote_path = f"{server['username']}@{server['hostname']}:{server['local_path']}"
        
        # SCP command to copy Google Drive folder to remote server
        scp_command = ["scp", "-rv", google_drive_path, remote_path]
        
        try:
            result = subprocess.run(scp_command, capture_output=True, text=True, check=True)
            if result.returncode == 0:
                print(f"Google Drive folder successfully copied to {server['hostname']}")
        except subprocess.CalledProcessError as e:
            print(f"Error copying to {server['hostname']}: {e}")

# Run the copy function
copy_google_drive_scp()
