#!/usr/bin/env python3
import os
import sys
import time
import shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN



def print_progress(current_step, total_steps, start_time):
    """
    Print a progress message showing:
      1) How many steps have been completed / how many remain
      2) Elapsed time
      3) Estimated remaining time to complete ALL steps in the pipeline
    """
    import time
    elapsed = time.time() - start_time
    progress_fraction = current_step / total_steps if total_steps > 0 else 1
    steps_left = total_steps - current_step

    if progress_fraction > 0:
        estimated_total_time = elapsed / progress_fraction
    else:
        estimated_total_time = 0.0

    remaining = estimated_total_time - elapsed
    print(f"[Step {current_step}/{total_steps}] "
          f"({steps_left} steps left) "
          f"Elapsed time: {elapsed:.1f}s | "
          f"Estimated remaining time: {remaining:.1f}s")


def get_unique_filename(base_name, extension, output_dir='.'):
    """
    Generate a unique filename inside 'output_dir' by appending an integer.
    E.g. predicted_pockets_1.pdb, predicted_pockets_2.pdb, etc.
    """
    i = 1
    while True:
        filename = os.path.join(output_dir, f"{base_name}_{i}.{extension}")
        if not os.path.exists(filename):
            return filename
        i += 1

# ------------------------------------------------------------------------------
# Parsing PDB
# ------------------------------------------------------------------------------
def read_pdb_atoms(filename):
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                res_name  = line[17:20].strip()
                chain     = line[21].strip()
                res_num   = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atoms.append({
                    'atom_name': atom_name,
                    'res_name':  res_name,
                    'chain':     chain,
                    'res_num':   res_num,
                    'x':         x,
                    'y':         y,
                    'z':         z
                })
    return atoms

# ------------------------------------------------------------------------------
# Creating a 3D grid
# ------------------------------------------------------------------------------
def create_grid(coords, spacing=0.5):
    x_min, y_min, z_min = coords.min(axis=0) - 5
    x_max, y_max, z_max = coords.max(axis=0) + 5
    grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:spacing,
                                      y_min:y_max:spacing,
                                      z_min:z_max:spacing]
    return np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

# ------------------------------------------------------------------------------
# Difference of Gaussians (DoG)
# ------------------------------------------------------------------------------
def apply_dog_filter(grid, protein_coords, sigma1=1.0, sigma2=2.0):
    density = np.zeros(len(grid))
    for coord in protein_coords:
        dist = np.linalg.norm(grid - coord, axis=1)
        density += np.exp(-dist**2 / (2 * sigma1**2))

    blurred1 = gaussian_filter(density, sigma=sigma1)
    blurred2 = gaussian_filter(density, sigma=sigma2)
    dog_result = blurred1 - blurred2

    # Normalize 0..1
    min_val, max_val = np.min(dog_result), np.max(dog_result)
    return (dog_result - min_val) / (max_val - min_val) if max_val != min_val else dog_result

# ------------------------------------------------------------------------------
# Extract top DoG points
# ------------------------------------------------------------------------------
def extract_pocket_points(grid, dog_filtered, threshold_percentile=95):
    threshold = np.percentile(dog_filtered, threshold_percentile)
    return grid[dog_filtered > threshold]

# ------------------------------------------------------------------------------
# Cluster pockets (DBSCAN)
# ------------------------------------------------------------------------------
def cluster_pockets(pocket_points, eps=0.8, min_samples=5):
    if len(pocket_points) < 2:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
    labels = clustering.labels_

    clustered_pockets = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # ignore noise

    MAX_CLUSTER_VOLUME = 3000.0
    for cluster_id in unique_labels:
        coords = pocket_points[labels == cluster_id]
        if len(coords) < 5:
            continue

        centroid = coords.mean(axis=0)
        hull = ConvexHull(coords)
        volume, surface_area = hull.volume, hull.area
        dogsite_score = min((volume / (surface_area + 1e-6)) * 10, 1.0)

        if volume > MAX_CLUSTER_VOLUME:
            continue

        clustered_pockets.append({
            'cluster_id': cluster_id,
            'coords': coords,
            'centroid': centroid,
            'volume': volume,
            'surface_area': surface_area,
            'dogsite_score': dogsite_score
        })

    return clustered_pockets

# ------------------------------------------------------------------------------
# Saving pocket centroids in a PDB
# ------------------------------------------------------------------------------
def save_pockets_as_pdb(pockets, output_filename):
    with open(output_filename, 'w') as file:
        file.write("REMARK  Generated by Binding Site Predictor\n")
        file.write("REMARK  Columns:\n")
        file.write("REMARK  HETATM  ID  Residue  Chain  ResNum  X      Y      Z      Occupancy  Score  Volume   SurfaceArea\n")
        for i, pocket in enumerate(pockets):
            x, y, z = pocket['centroid']
            vol = pocket['volume']
            sa  = pocket['surface_area']
            score = pocket['dogsite_score']
            file.write(
                f"HETATM{i:5d}  POC POC A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  {score:.2f}  V={vol:.2f} SA={sa:.2f}\n"
            )

# ------------------------------------------------------------------------------
# Find which residues form each pocket (distance-based)
# ------------------------------------------------------------------------------
def find_residues_in_pocket(atom_list, centroid, distance_threshold=4.0):
    residues_in_pocket = set()
    for atom in atom_list:
        dx = atom['x'] - centroid[0]
        dy = atom['y'] - centroid[1]
        dz = atom['z'] - centroid[2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist <= distance_threshold:
            residues_in_pocket.add((atom['chain'], atom['res_name'], atom['res_num']))
    return sorted(residues_in_pocket, key=lambda x: (x[0], x[2]))

# ------------------------------------------------------------------------------
# Save only certain residues in a PDB
# ------------------------------------------------------------------------------
def save_residues_as_pdb(atom_list, residues, output_filename):
    residue_set = set(residues)
    with open(output_filename, 'w') as out:
        out.write("REMARK  Residues forming the pocket\n")
        out.write("REMARK  CHAIN, RESNAME, RESNUM\n")
        for r in residues:
            out.write(f"REMARK  {r[0]} {r[1]} {r[2]}\n")
        out.write("REMARK \n")

        atom_id = 1
        for atom in atom_list:
            c  = atom['chain']
            rn = atom['res_name']
            rnum = atom['res_num']
            if (c, rn, rnum) in residue_set:
                out.write(
                    f"ATOM  {atom_id:5d} {atom['atom_name']:^4s}"
                    f"{rn:>3s} {c}{rnum:4d}    "
                    f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00\n"
                )
                atom_id += 1

# ------------------------------------------------------------------------------
# Generate PyMOL script
# ------------------------------------------------------------------------------
def generate_pymol_script(original_pdb_abs, pockets_pdb_abs, output_script_abs):
    """
    We use absolute paths so PyMOL can open them from any working directory.
    """
    with open(output_script_abs, 'w') as file:
        file.write(f"load {original_pdb_abs}\n")
        file.write("surface\n")
        file.write("color grey80, all\n")
        file.write(f"load {pockets_pdb_abs}\n")
        file.write("show spheres, resn POC\n")
        file.write("color yellow, resn POC\n")
        file.write("set sphere_scale, 1.0\n")
        file.write("zoom all\n")

# ------------------------------------------------------------------------------
# Generate Chimera script
# ------------------------------------------------------------------------------
def generate_chimera_script(original_pdb_abs, pockets_pdb_abs,
                            pocket_residues_abs_list, output_script_abs):
    """
    Also uses absolute paths so that Chimera will find the files no matter
    where you open the .cmd script from.
    """
    colors = ["red", "green", "blue", "magenta", "cyan", "orange"]
    with open(output_script_abs, 'w') as file:
        # load the original protein
        file.write(f"open {original_pdb_abs}\n")
        file.write("surface #0\n")
        file.write("surfrepr mesh #0\n\n")

        # load the pockets (centroids)
        file.write(f"open {pockets_pdb_abs}\n")
        file.write("select #1 & :POC\n")
        file.write("rep sphere sel\n")
        file.write("color yellow sel\n")
        file.write("~select\n")
        file.write("focus\n\n")

        # load the pocket residue files
        for i, residue_pdb_abs in enumerate(pocket_residues_abs_list, start=2):
            color_to_use = colors[(i - 2) % len(colors)]
            file.write(f"open {residue_pdb_abs}\n")
            file.write(f"color {color_to_use} #{i}\n")
            file.write(f"surface #{i}\n\n")

        file.write("focus\n")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 detect_pockets_time.py <protein_structure.pdb>")
        sys.exit(1)

    input_pdb_filename = sys.argv[1]
    if not os.path.isfile(input_pdb_filename):
        print(f"ERROR: File {input_pdb_filename} does not exist.")
        sys.exit(1)

    # Create results folder named after the PDB
    pdb_basename = os.path.splitext(os.path.basename(input_pdb_filename))[0]
    results_dir = pdb_basename + "_results"
    os.makedirs(results_dir, exist_ok=True)

    # Copy the input PDB to that folder
    shutil.copy(input_pdb_filename, results_dir)

    # We'll also prepare a *absolute path* for that copied PDB
    original_pdb_in_results = os.path.join(results_dir, os.path.basename(input_pdb_filename))
    original_pdb_abs = os.path.abspath(original_pdb_in_results)

    # Steps for the pipeline
    TOTAL_STEPS = 7
    current_step = 0
    start_time = time.time()

    # 1) read atoms
    protein_atoms = read_pdb_atoms(input_pdb_filename)
    coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
    current_step += 1
    print_progress(current_step, TOTAL_STEPS, start_time)

    # 2) create grid
    grid_points = create_grid(coords, spacing=0.5)
    current_step += 1
    print_progress(current_step, TOTAL_STEPS, start_time)

    # 3) apply DoG
    dog_filtered = apply_dog_filter(grid_points, coords)
    current_step += 1
    print_progress(current_step, TOTAL_STEPS, start_time)

    # 4) extract points
    pocket_candidates = extract_pocket_points(grid_points, dog_filtered, threshold_percentile=95)
    current_step += 1
    print_progress(current_step, TOTAL_STEPS, start_time)

    # 5) cluster pockets
    pocket_clusters = cluster_pockets(pocket_candidates)
    current_step += 1
    print_progress(current_step, TOTAL_STEPS, start_time)

    # 6) Save pockets & residue files, generate scripts
    #    All in results_dir, but we also store absolute paths for the script references
    pockets_pdb_path     = get_unique_filename("predicted_pockets", "pdb", results_dir)
    pymol_script_path    = get_unique_filename("visualize_pockets", "pml", results_dir)
    chimera_script_path  = get_unique_filename("visualize_pockets", "cmd", results_dir)

    pockets_pdb_abs       = os.path.abspath(pockets_pdb_path)
    pymol_script_abs      = os.path.abspath(pymol_script_path)
    chimera_script_abs    = os.path.abspath(chimera_script_path)

    pocket_residues_abs_list = []

    if len(pocket_clusters) > 0:
        # save the pocket centroids
        save_pockets_as_pdb(pocket_clusters, pockets_pdb_path)

        # for each pocket cluster, find residues
        for i, pocket in enumerate(pocket_clusters, start=1):
            residues = find_residues_in_pocket(protein_atoms, pocket['centroid'], distance_threshold=5.0)
            res_pdb_filename = get_unique_filename(f"pocket_{i}_residues", "pdb", results_dir)
            save_residues_as_pdb(protein_atoms, residues, res_pdb_filename)
            pocket_residues_abs_list.append(os.path.abspath(res_pdb_filename))

        # generate PyMOL & Chimera scripts referencing absolute paths
        generate_pymol_script(original_pdb_abs, pockets_pdb_abs, pymol_script_abs)
        generate_chimera_script(original_pdb_abs, pockets_pdb_abs, pocket_residues_abs_list, chimera_script_abs)

        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)

        print(f"✅ Pockets file saved as: {pockets_pdb_path}")
        print(f"✅ PyMOL script saved as: {pymol_script_path}")
        print(f"✅ Chimera script saved as: {chimera_script_path}")
    else:
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        print("❌ No significant pockets found.")

    # 7) done
    print("Processing complete.")

