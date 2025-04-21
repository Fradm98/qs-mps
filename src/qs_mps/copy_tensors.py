import numpy as np
import subprocess
from qs_mps.utils import get_cx, get_cy

# Source servers to copy from
source_servers = ["fradm@marcos1", "fradm@marcos2"]

# Destination server and folder
dest_server = "fradm@marcos3"
remote_dir = "/Users/fradm/Desktop/projects/1_Z2/results/tensors/"

L, l = 30, 6
Rs = [10,11,12,13,14,15,16,17,18,19,20]
h_i, h_f, npoints = 0.4, 0.6, 21
h_i, h_f, npoints = 0.6, 0.9, 31
precision = 3
chi = 64
bc = "pbc"
R = 0
cx = get_cx(L, R)
cy = get_cy(l, bc, R=R)
tensor_files_vac = [f"tensor_sites_Z2_dual_direct_lattice_6x30_bc_pbc_vacuum_sector_'{cx}'-'{cy}'_chi_{chi}_h_{g:.{precision}f}.h5" for g in np.linspace(h_i,h_f,npoints)]


import paramiko
import os
from scp import SCPClient, SCPException
from datetime import datetime

# Choose the observable and tensor list
device = "pc"  # or "pc", "mac", "marcos1", etc.
observable = "tensors"

# List of server information
all_servers = [
    # {"hostname": "158.227.6.203", "username": "fradm", "key_filename": None},  # marcos1
    # {"hostname": "158.227.46.38", "username": "fradm", "key_filename": None},   # marcos2
    {"hostname": "158.227.47.136", "username": "fradm", "key_filename": None},  # marcos3
]

# Local and remote result directories
remote_results_dir = f"/Users/fradm/Desktop/projects/1_Z2/results/{observable}"

if device == "pc":
    key_filename = "C:/Users/HP/.ssh/id_rsa_marcos"
    local_results_dir = f"C:/Users/HP/Desktop/projects/1_Z2/results/{observable}"
    local_results_dir = f"D:/code/projects/1_Z2/results/{observable}"
elif device == "mac":
    key_filename = "/Users/fradm98/.ssh/id_rsa_mac"
    local_results_dir = f"/Users/fradm98/Desktop/projects/1_Z2/results/{observable}"
elif device == "marcos1":
    key_filename = "/Users/fradm/.ssh/id_rsa_marcos1"
    local_results_dir = remote_results_dir
    all_servers.pop(0)
elif device == "marcos2":
    key_filename = "/Users/fradm/.ssh/id_rsa_marcos2"
    local_results_dir = remote_results_dir
    all_servers.pop(1)
elif device == "marcos3":
    key_filename = "/Users/fradm/.ssh/id_rsa_marcos3"
    local_results_dir = remote_results_dir
    all_servers.pop(2)

servers = []
for server in all_servers:
    server["key_filename"] = key_filename
    servers.append(server)


def get_remote_files(client, remote_dir, tensor_files):
    # Returns a filtered list of tensor files in remote directory
    stdin, stdout, stderr = client.exec_command(f"ls -lTot {remote_dir}")
    files = {}

    for line in stdout:
        parts = line.strip().split()

        # Skip lines that don't correspond to files
        if len(parts) < 9 or parts[0][0] == "d":
            continue  # Ignore directories and malformed lines
        else:
            filename = " ".join(parts[8:])  # Join any spaces in the filename

            if filename in tensor_files:  # Only process tensor files
                # Parse the date string to a timestamp
                date_str = f"{parts[4]} {parts[5]} {parts[6]} {parts[7]}"
                date_format = "%b %d %H:%M:%S %Y"
                try:
                    timestamp = int(datetime.strptime(date_str, date_format).timestamp())
                except ValueError as e:
                    print(f"Error parsing date: {date_str}. {e}")
                    continue  # Skip files with parsing issues

                files[f"{remote_dir}/{filename}"] = timestamp

    return files


def sync_files(server, tensor_files, max_attempts=3):
    print(f"Syncing files from {server['hostname']}...")
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

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

    remote_files = get_remote_files(client, remote_results_dir, tensor_files)

    # SCP transfer
    with SCPClient(client.get_transport(), socket_timeout=180) as scp:
        files_modified = 0
        for remote_file, remote_timestamp in remote_files.items():
            local_file = f"{local_results_dir}/{os.path.basename(remote_file)}"

            if os.path.exists(local_file):
                # Get the local file's modification time
                local_timestamp = int(os.path.getmtime(local_file))

                # Compare timestamps
                if remote_timestamp <= local_timestamp:
                    continue  # Skip copying if local file is newer or equal
                else:
                    print(f"Local file is older: {os.path.basename(remote_file)}. Copying the newer remote file")
                    scp.get(remote_file, local_file)
                    files_modified += 1
            else:
                print(f"Local file does not exist. Copying the remote file")
                try:
                    scp.get(remote_file, local_results_dir)
                    files_modified += 1
                    print(f"Copied: {os.path.basename(remote_file)}")
                except SCPException as e:
                    print(f"Error copying file: {e}")

        if files_modified == 0:
            print(f"All files are up-to-date")
        else:
            print(f"{files_modified} files have been modified and/or added")

    client.close()


# Sync files from each server
for server in servers:
    for R in Rs:
        cx = get_cx(L, R)
        cy = get_cy(l, bc, R=R)
        # Tensor filenames you want to copy
        tensor_files_R = [
            f"tensor_sites_Z2_dual_direct_lattice_6x30_bc_pbc_2_particle(s)_sector_{cx}-{cy}_chi_{chi}_h_{g:.{precision}f}.h5" for g in np.linspace(h_i,h_f,npoints)
        ]

        sync_files(server, tensor_files_R)
    sync_files(server, tensor_files_vac)
