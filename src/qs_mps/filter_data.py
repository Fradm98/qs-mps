import os
import pandas as pd
from qs_mps.utils import get_cx
import shutil

# Define the folder where your data is stored
destination_folder = "C:/Users/HP/Desktop/vanilla_data/results/energy_density_data"
data_folder = "C:/Users/HP/Desktop/projects/1_Z2/results/energy_data"

destination_folder = "/Users/fradm/Desktop/vanilla_data/results/kink_mass_data"
data_folder = "/Users/fradm/Desktop/projects/1_Z2/results/energy_data"

L = 30
cy = [0,0]
bc = "pbc"
sector = "2_particle(s)_sector"

d = dict(l=5,f="tua_mamma")

# params = dict(
#     tab1 = dict(
#         l = 6,
#         h_i = 0.6,
#         h_f = 0.95,
#         npoints = 15,
#         chis = [32,64,128,256],
#         Rs = ["vac",7,8,9,10,11,12,13,14,15,16,17,19,21]
# ),
#     tab2 = dict(
#         l = 6,
#         h_i = 0.6, 
#         h_f = 0.8, 
#         npoints = 30,
#         chis = [16,32,64,128],
#         Rs = ["vac",10,12,14,16,18,20,22,24]
# ),
#     tab3 = dict(
#         l = 6,
#         h_i = 0.4, 
#         h_f = 5.0, 
#         npoints = 33,
#         chis = [16,32,64,128],
#         Rs = ["vac",14,16,18,20]
# ),
#     tab4 = dict(
#         l = 5,
#         h_i = 0.4, 
#         h_f = 1.0, 
#         npoints = 61,
#         chis = [16,32,64,128],
#         Rs = ["vac",18,20,22,24]
# ),
#     tab5 = dict(
#         l = 5,
#         h_i = 0.6, 
#         h_f = 0.9, 
#         npoints = 31,
#         chis = [32,64,128,256],
#         Rs = ["vac",7,8,9,10,11,13,15,17,19,21]
# ),
#     tab6 = dict(
#         l = 4,
#         h_i = 0.4, 
#         h_f = 1.0, 
#         npoints = 61,
#         chis = [16,32,64,128],
#         Rs = ["vac",18,20,22,24]
# ),
#     tab7 = dict(
#         l = 4,
#         h_i = 0.6, 
#         h_f = 0.9, 
#         npoints = 31,
#         chis = [32,64,128,256],
#         Rs = ["vac",7,8,9,10,11,12,14,16,18,20,22]
# ))

params = dict(
    tab1 = dict(
        l = 6,
        h_i = 0.6,
        h_f = 0.95,
        npoints = 15,
        chis = [64,128],
        Rs = ["vac",10,11,12,13,14,15,16,17,19]
),
    tab2 = dict(
        l = 5,
        h_i = 0.2, 
        h_f = 1.0, 
        npoints = 21,
        chis = [64,128],
        Rs = ["vac",10,11,12,13,14,15,16,17,18,19,20]
),
    tab3 = dict(
        l = 5,
        h_i = 1.0, 
        h_f = 2.0, 
        npoints = 11,
        chis = [64,128],
        Rs = ["vac",10,11,12,13,14,15,16,17,18,19,20]
),
    tab4 = dict(
        l = 5,
        h_i = 0.8, 
        h_f = 1.0, 
        npoints = 41,
        chis = [64,128],
        Rs = ["vac",10,11,12,13,14,15,16,17,18,19,20]
))

for key, param in params.items():
    # Create an empty DataFrame
    table = pd.DataFrame(index=param["chis"], columns=param["Rs"])

    # Iterate over all combinations of sweep variables
    for chi in param['chis']:
        for R in param['Rs']:
            if R == "vac":
                sector = "vacuum_sector"
                cx = None
                cy = None
            else:
                sector = "2_particle(s)_sector"
                cx = get_cx(L,R)
                cy = [0,0]
            
            # Construct the expected filename pattern
            # filename = f"electric_energy_density_Z2_dual_direct_lattice_{param["l"]}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{param["h_i"]}-{param["h_f"]}_delta_{param["npoints"]}_chi_{chi}.npy"  # Adjust extension if needed
            filename = f"energy_Z2_dual_direct_lattice_{param['l']}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{param['h_i']}-{param['h_f']}_delta_{param['npoints']}_chi_{chi}.npy"  # Adjust extension if needed
            
            # Check if the file exists
            file_exists = os.path.exists(os.path.join(data_folder, filename))
            
            # Fill the table with "yes" or "no"
            table.loc[chi, R] = "yes" if file_exists else "no"
            if file_exists:
                file_path = os.path.join(data_folder, filename)
                # Copy the file to the destination folder
                shutil.copy(file_path, os.path.join(destination_folder, filename))

    # Print the table
    print(f"l: {param['l']}, h_i: {param['h_i']}, h_f: {param['h_f']}, npoints: {param['npoints']}")
    print(table)

    # Optionally, save to a CSV file
    table.to_csv(f"/Users/fradm/Desktop/vanilla_data/results/output_table_on_axis_{param['l']}x{L}_h_{param['h_i']}-{param['h_f']}_npoints_{param['npoints']}.csv")

    # Iterate over all combinations of sweep variables
    for chi in param['chis']:
        for R in param['Rs']:
            if R == "vac":
                sector = "vacuum_sector"
                cx = None
                cy = None
            else:
                sector = "2_particle(s)_sector"
                cx = get_cx(L,R)
                cy = [0,1]
            
            # Construct the expected filename pattern
            # filename = f"electric_energy_density_Z2_dual_direct_lattice_{param["l"]}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{param["h_i"]}-{param["h_f"]}_delta_{param["npoints"]}_chi_{chi}.npy"  # Adjust extension if needed
            filename = f"energy_Z2_dual_direct_lattice_{param['l']}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{param['h_i']}-{param['h_f']}_delta_{param['npoints']}_chi_{chi}.npy"  # Adjust extension if needed
            
            # Check if the file exists
            file_exists = os.path.exists(os.path.join(data_folder, filename))
            
            # Fill the table with "yes" or "no"
            table.loc[chi, R] = "yes" if file_exists else "no"
            if file_exists:
                file_path = os.path.join(data_folder, filename)
                # Copy the file to the destination folder
                shutil.copy(file_path, os.path.join(destination_folder, filename))

    # Print the table
    print(f"l: {param['l']}, h_i: {param['h_i']}, h_f: {param['h_f']}, npoints: {param['npoints']}")
    print(table)

    # Optionally, save to a CSV file
    table.to_csv(f"/Users/fradm/Desktop/vanilla_data/results/output_table_off_axis_{param['l']}x{L}_h_{param['h_i']}-{param['h_f']}_npoints_{param['npoints']}.csv")

