import paramiko
import os
from scp import SCPClient, SCPException
from datetime import datetime

# List of server information
servers = [
    {"hostname": "158.227.6.203", "username": "fradm", "key_filename": "C:/Users/HP/.ssh/id_rsa_marcos"},
    {"hostname": "158.227.46.38", "username": "fradm", "key_filename": "C:/Users/HP/.ssh/id_rsa_marcos"},
    {"hostname": "158.227.47.136", "username": "fradm", "key_filename": "C:/Users/HP/.ssh/id_rsa_marcos"},
]

# Local and remote result directories
local_results_dir = "C:/Users/HP/Desktop/projects/1_Z2/results/energy_data"
remote_results_dir = "/Users/fradm/Desktop/projects/1_Z2/results/energy_data"

def get_remote_files(client, remote_dir):
    stdin, stdout, stderr = client.exec_command(f"ls -l {remote_dir}")
    files = {}

    for line in stdout:
        parts = line.strip().split()
        
        # Skip lines that don't correspond to files
        if len(parts) < 9 or parts[0][0] == 'd':
            continue  # Ignore directories and malformed lines
        
        # Extract the filename and modification time
        filename = ' '.join(parts[8:])  # Join any spaces in the filename
        date_str = ' '.join(parts[5:8])  # Join the date components
        current_year = datetime.now().year
        date_str += f" {current_year}"
        # Parse the date string to a timestamp
        try:
            timestamp = int(datetime.strptime(date_str, '%b %d %H:%M %Y').timestamp())
        except ValueError as e:
            print(f"Error parsing date: {date_str}. {e}")
            continue  # Skip files with parsing issues

        # Add the full path and timestamp to the dictionary
        files[f"{remote_dir}/{filename}"] = timestamp
    
    return files


def sync_files(server, max_attempts=3):
    print(f"Syncing files from {server['hostname']}...")
    client = paramiko.SSHClient()
    for attempt in range(max_attempts):
        try:
            client.connect(
                hostname=server["hostname"],
                username=server["username"],
                key_filename=server["key_filename"],
                timeout=180,
            )
            break  # Exit the loop if the connection is successful
        except (paramiko.SSHException, TimeoutError) as e:
            print(f"Attempt {attempt + 1} failed to connect: {e}")
            if attempt < max_attempts - 1:
                print("Retrying...")
            else:
                print("Max attempts reached. Skipping this server.")
                return  # Exit the function if all attempts fail

    remote_files = get_remote_files(client, remote_results_dir)

    # SCP transfer
    with SCPClient(client.get_transport(), socket_timeout=180) as scp:
        for remote_file, remote_timestamp in remote_files.items():
            local_file = f"{local_results_dir}/{os.path.basename(remote_file)}"

            if os.path.exists(local_file):
                # Get the local file's modification time
                local_timestamp = int(os.path.getmtime(local_file))

                # Compare timestamps
                if remote_timestamp <= local_timestamp:
                    print(f"File is up-to-date: {local_file}. Skipping.")
                    continue  # Skip copying if local file is newer or equal
                else:
                    print(f"Local file is older: {local_file}. Copying the newer remote file")
                    scp.get(remote_file, local_file)
            else:
                print(f"Local file does not exist. Copying the remote file")
                try:
                    # Replace backslashes with forward slashes for the remote path
                    scp.get(remote_file, local_results_dir)
                    print(f"Copied: {local_file}")
                except SCPException as e:
                    print(f"Error copying file: {e}")

    client.close()


# Sync files from each server
for server in servers:
    sync_files(server)




# import subprocess

# # Define server details
# servers = [
#     {"hostname": "158.227.6.203", "username": "fradm", "local_path": "Desktop/projects/1_Z2/results"},
#     {"hostname": "158.227.46.38", "username": "fradm", "local_path": "Desktop/projects/1_Z2/results"},
#     {"hostname": "158.227.47.136", "username": "fradm", "local_path": "Desktop/projects/1_Z2/results"}
# ]

# # Define the local directory to sync to
# local_dir = "C:/Users/HP/Desktop/projects/1_Z2/results"



# # Function to generate a filename or search pattern based on parameters
# def generate_filename(params):
#     filename = "_".join(f"{key}_{value}" for key, value in params.items())
#     return filename

# # Use rsync to fetch result from any server where it exists
# def fetch_result(params):
#     filename_pattern = generate_filename(params)
#     for server in servers:
#         remote_path = f"{server['username']}@{server['hostname']}:{server['local_path']}/{filename_pattern}*"
        
#         # Rsync command to sync matching files to the local directory
#         rsync_command = ["rsync", "-avz", "--ignore-missing-args", remote_path, local_dir]
        
#         try:
#             result = subprocess.run(rsync_command, capture_output=True, text=True, check=True)
#             if result.returncode == 0 and "skipping non-regular file" not in result.stdout:
#                 print(f"Result synced from {server['hostname']}")
#                 return f"{local_dir}/{filename_pattern}"
#         except subprocess.CalledProcessError as e:
#             print(f"Error syncing from {server['hostname']}: {e}")
    
#     print("Result not found on any server. Computation may still be required.")
#     return None


# # Local path of the Google Drive folder to copy
# google_drive_path = "G:/My Drive/projects/1_Z2/results/"

# # Function to copy Google Drive folder to each server using scp
# def copy_google_drive_scp():
#     for server in servers:
#         remote_path = f"{server['username']}@{server['hostname']}:{server['local_path']}"
        
#         # SCP command to copy Google Drive folder to remote server
#         scp_command = ["scp", "-rv", google_drive_path, remote_path]
        
#         try:
#             result = subprocess.run(scp_command, capture_output=True, text=True, check=True)
#             if result.returncode == 0:
#                 print(f"Google Drive folder successfully copied to {server['hostname']}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error copying to {server['hostname']}: {e}")

# # Run the copy function
# copy_google_drive_scp()
