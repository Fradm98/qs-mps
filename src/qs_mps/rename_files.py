import os

# Define the directory containing the files
directory = "/Users/fradm/Desktop/projects/1_Z2/results/tensors"

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the filename contains both "vacuum_sector" and "[]"
    if "vacuum_sector" in filename and "[]" in filename:
        # Construct the new filename
        new_filename = filename.replace("vacuum_sector", "2_particle(s)_sector").replace("[]", "[0, 0]")
        
        # Full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

# for filename in os.listdir(directory):
#     # Check if the filename contains both "vacuum_sector" and "[]"
#     if "2_particles_sector" in filename:
#         # Construct the new filename
#         new_filename = filename.replace("2_particles_sector", "2_particle(s)_sector")
        
#         # Full paths
#         old_path = os.path.join(directory, filename)
#         new_path = os.path.join(directory, new_filename)

#         # Rename the file
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} -> {new_filename}")