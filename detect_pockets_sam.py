#!/usr/bin/env python3
import os
import sys
import shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

def get_unique_filename(folder_path, base_name, extension):
    """
    Generates a unique filename (base_name_1.extension, base_name_2.extension, etc.)
    inside the specified folder_path.
    """
    i = 1
    while True:
        filename = os.path.join(folder_path, f"{base_name}_{i}.{extension}")
        if not os.path.exists(filename):
            return filename
        i += 1

# ------------------------------------------------------------------------------
# 1) Read PDB atoms
# ------------------------------------------------------------------------------
def read_pdb_atoms(filename):
    """
    Reads a PDB file and returns a list of dictionaries with atom information:
    {
        'atom_name': str,
        'res_name': str,
        'chain': str,
        'res_num': int,
        'x': float,
        'y': float,
        'z': float
    }
    Only considers lines starting with 'ATOM'.
    """
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

def read_pdb_coords(filename):
    """
    Extracts atomic coordinates from a PDB file (only x, y, z).
    Considers lines starting with 'ATOM'.
    """
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

# ------------------------------------------------------------------------------
# 2) Create a 3D grid around the protein
# ------------------------------------------------------------------------------
def create_grid(coords, spacing=0.5):
    """
    Generates a 3D grid around the given coordinates, with a 5 Å border.
    """
    x_min, y_min, z_min = coords.min(axis=0) - 5
    x_max, y_max, z_max = coords.max(axis=0) + 5
    grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:spacing,
                                      y_min:y_max:spacing,
                                      z_min:z_max:spacing]
    return np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

# ------------------------------------------------------------------------------
# 3) Apply Difference of Gaussians (DoG)
# ------------------------------------------------------------------------------
def apply_dog_filter(grid, protein_coords, sigma1=1.0, sigma2=2.0):
    """
    Applies a Difference of Gaussians operation to detect potential pockets.
    Not a Biopython predictor, just a mathematical approach.
    """
    density = np.zeros(len(grid))
    for coord in protein_coords:
        dist = np.linalg.norm(grid - coord, axis=1)
        density += np.exp(-dist**2 / (2 * sigma1**2))

    blurred1 = gaussian_filter(density, sigma=sigma1)
    blurred2 = gaussian_filter(density, sigma=sigma2)
    dog_result = blurred1 - blurred2

    # Normalize between 0 and 1
    return (dog_result - np.min(dog_result)) / (np.max(dog_result) - np.min(dog_result))

# ------------------------------------------------------------------------------
# 4) Extract grid points above a DoG threshold
# ------------------------------------------------------------------------------
def extract_pocket_points(grid, dog_filtered, threshold_percentile=95):
    """
    Returns the grid points whose DoG value is above a certain percentile.
    """
    threshold = np.percentile(dog_filtered, threshold_percentile)
    return grid[dog_filtered > threshold]

# ------------------------------------------------------------------------------
# 5) Cluster pockets with DBSCAN
# ------------------------------------------------------------------------------
def cluster_pockets(pocket_points, eps=0.8, min_samples=5):
    """
    Clusters 'pocket_points' using DBSCAN, returning a list of pocket dictionaries.
    """
    if len(pocket_points) < 2:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
    labels = clustering.labels_

    clustered_pockets = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Noise

    # Volume filter for extremely large clusters
    MAX_CLUSTER_VOLUME = 3000.0

    for cluster_id in unique_labels:
        cluster_coords = pocket_points[labels == cluster_id]
        if len(cluster_coords) < 5:
            continue

        centroid = cluster_coords.mean(axis=0)
        hull = ConvexHull(cluster_coords)
        volume, surface_area = hull.volume, hull.area
        dogsite_score = min((volume / (surface_area + 1e-6)) * 10, 1.0)

        # Discard pockets with excessive volume
        if volume > MAX_CLUSTER_VOLUME:
            continue

        clustered_pockets.append({
            'cluster_id': cluster_id,
            'coords': cluster_coords,
            'centroid': centroid,
            'volume': volume,
            'surface_area': surface_area,
            'dogsite_score': dogsite_score
        })

    return clustered_pockets

# ------------------------------------------------------------------------------
# 6) Save pocket centroids in a PDB file
# ------------------------------------------------------------------------------
def save_pockets_as_pdb(pockets, output_filename):
    """
    Saves the predicted pockets in a PDB file with a descriptive header.
    Each pocket is stored as a HETATM line (coordinates = centroid).
    """
    with open(output_filename, 'w') as file:
        file.write("REMARK  Generated by Binding Site Predictor\n")
        file.write("REMARK  Columns:\n")
        file.write("REMARK  HETATM  ID  Residue  Chain  ResNum  X      Y      Z      Occupancy  Score  Volume   SurfaceArea\n")

        for i, pocket in enumerate(pockets):
            x, y, z = pocket['centroid']
            volume = pocket['volume']
            surface_area = pocket['surface_area']
            dogsite_score = pocket['dogsite_score']

            file.write(
                f"HETATM{i:5d}  POC POC A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  {dogsite_score:.2f}  V={volume:.2f} SA={surface_area:.2f}\n"
            )

# ------------------------------------------------------------------------------
# 6b) Find residues in the pocket (distance-based)
# ------------------------------------------------------------------------------
def find_residues_in_pocket(atom_list, centroid, distance_threshold=5.0):
    """
    Finds residues within 'distance_threshold' Å of the pocket centroid.
    Returns a sorted list of (chain, res_name, res_num).
    """
    residues_in_pocket = set()
    for atom in atom_list:
        x, y, z = atom['x'], atom['y'], atom['z']
        dist = np.linalg.norm(np.array([x, y, z]) - centroid)
        if dist <= distance_threshold:
            residues_in_pocket.add((atom['chain'], atom['res_name'], atom['res_num']))

    # Sort by chain and residue number
    return sorted(residues_in_pocket, key=lambda x: (x[0], x[2]))

# ------------------------------------------------------------------------------
# 6c) Save only certain residues (atoms) in a new PDB
# ------------------------------------------------------------------------------
def save_residues_as_pdb(atom_list, residues, output_filename):
    """
    Saves a PDB file containing only atoms belonging to the given 'residues',
    where each residue is (chain, res_name, res_num).
    """
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
# 7) Visualization scripts (PyMOL and Chimera)
# ------------------------------------------------------------------------------
def generate_pymol_script(
    original_pdb_name,
    pockets_pdb_name,
    pocket_residues_names,
    output_script
):
    """
    Creates a PyMOL script that:
      - Opens the original protein
      - Opens the pocket centroids (pockets_pdb_name) and shows them as a transparent surface
      - Opens each pocket residue file, also as a transparent surface
    """
    with open(output_script, 'w') as file:
        # Load the original protein
        file.write(f"load {original_pdb_name}\n")
        file.write("show cartoon\n")
        file.write("color gray, all\n")
        file.write("zoom\n")
        file.write("\n")

        # Pocket centroids as surface with transparency
        file.write(f"load {pockets_pdb_name}\n")
        file.write("show surface, resn POC\n")
        file.write("color yellow, resn POC\n")
        file.write("set transparency, 0.5, resn POC\n")
        file.write("\n")

        # Each pocket residue file as surface with transparency
        colors = ["red", "green", "blue", "magenta", "cyan", "orange"]
        for i, residue_pdb_name in enumerate(pocket_residues_names):
            color = colors[i % len(colors)]
            file.write(f"load {residue_pdb_name}\n")
            file.write("show surface\n")
            file.write(f"color {color}, (all)\n")
            file.write("set transparency, 0.5, (all)\n")
            file.write("\n")

        file.write("zoom\n")

def generate_chimera_script(
    original_pdb_name,
    pockets_pdb_name,
    pocket_residues_names,
    output_script
):
    """
    Creates a Chimera script that:
      - Opens the original protein as a surface (mesh)
      - Opens the pockets centroids as spheres
      - Opens each pocket residue file as a transparent surface
    """
    with open(output_script, 'w') as file:
        # Open the original protein (#0)
        file.write(f"open {original_pdb_name}\n")
        file.write("surface #0\n")
        file.write("surfrepr mesh #0\n")
        file.write("\n")

        # Open the pockets centroids (#1)
        file.write(f"open {pockets_pdb_name}\n")
        file.write("select #1 & :POC\n")
        file.write("rep sphere sel\n")
        file.write("color yellow sel\n")
        file.write("~select\n")
        file.write("focus\n")
        file.write("\n")

        # For each pocket residue file (#2, #3, etc.) show surface with transparency
        colors = ["red", "green", "blue", "magenta", "cyan", "orange"]
        for i, residue_pdb_name in enumerate(pocket_residues_names, start=2):
            color = colors[(i - 2) % len(colors)]
            file.write(f"open {residue_pdb_name}\n")
            file.write(f"surface #{i}\n")
            file.write(f"transparency 50 #{i}\n")
            file.write(f"color {color} #{i}\n")
            file.write("\n")

        file.write("focus\n")

# ------------------------------------------------------------------------------
# 8) MAIN EXECUTION FLOW
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 detect_binding_sites.py <protein_file.pdb>")
        sys.exit(1)

    input_pdb_filename = sys.argv[1]
    if not os.path.isfile(input_pdb_filename):
        print(f"Error: file {input_pdb_filename} does not exist.")
        sys.exit(1)

    # 1. Extract the protein name without .pdb extension
    protein_name = os.path.splitext(os.path.basename(input_pdb_filename))[0]

    # 2. Create a folder with the protein name
    if not os.path.isdir(protein_name):
        os.makedirs(protein_name)

    # Copy the original PDB into that folder
    local_pdb = os.path.join(protein_name, f"{protein_name}.pdb")
    shutil.copy(input_pdb_filename, local_pdb)

    # 3. Read atoms and coords
    protein_atoms = read_pdb_atoms(input_pdb_filename)
    protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])

    # 4. Create the grid and apply DoG
    grid_points = create_grid(protein_coords, spacing=0.5)
    dog_filtered = apply_dog_filter(grid_points, protein_coords)

    # 5. Extract points and cluster
    pocket_candidates = extract_pocket_points(grid_points, dog_filtered, threshold_percentile=95)
    pocket_clusters = cluster_pockets(pocket_candidates, eps=0.8, min_samples=5)

    if len(pocket_clusters) == 0:
        print("❌ No significant pockets found.")
        sys.exit(0)

    # Generate unique filenames inside the new folder
    pockets_pdb = get_unique_filename(protein_name, "predicted_pockets", "pdb")
    pymol_script = get_unique_filename(protein_name, "visualize_pockets", "pml")
    chimera_script = get_unique_filename(protein_name, "visualize_pockets", "cmd")

    # 6. Save centroids to PDB
    save_pockets_as_pdb(pocket_clusters, pockets_pdb)
    print(f"✅ Pocket centroids saved as {pockets_pdb}")

    # 7. For each pocket, find and save residues
    pocket_residues_list = []
    for i, pocket in enumerate(pocket_clusters, start=1):
        residues = find_residues_in_pocket(protein_atoms, pocket['centroid'], distance_threshold=5.0)
        pocket_residues_pdb = get_unique_filename(protein_name, f"pocket_{i}_residues", "pdb")
        save_residues_as_pdb(protein_atoms, residues, pocket_residues_pdb)
        pocket_residues_list.append(pocket_residues_pdb)
        print(f"   Pocket {i} residues saved in: {pocket_residues_pdb}")

    # 8. For PyMOL and Chimera scripts, we only want the base filenames (no folder path)
    pockets_pdb_name     = os.path.basename(pockets_pdb)
    local_pdb_name       = os.path.basename(local_pdb)
    pymol_script_name    = os.path.basename(pymol_script)
    chimera_script_name  = os.path.basename(chimera_script)
    pocket_residues_names = [os.path.basename(x) for x in pocket_residues_list]

    # Generate the scripts referencing only the base filenames
    generate_pymol_script(local_pdb_name, pockets_pdb_name, pocket_residues_names, pymol_script)
    print(f"✅ PyMOL script saved as {pymol_script}")

    generate_chimera_script(local_pdb_name, pockets_pdb_name, pocket_residues_names, chimera_script)
    print(f"✅ Chimera script saved as {chimera_script}")

    print("\n📝 NOTE: Please 'cd' into the folder", protein_name,
          "before running the .cmd or .pml scripts to avoid file not found errors.")
