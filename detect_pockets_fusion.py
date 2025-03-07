#!/usr/bin/env python3
import os
import sys
import time
import shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

##############################
#  Helper Functions
##############################
def print_progress(current_step, total_steps, start_time):
    """
    Prints a progress message showing:
      1) How many steps have been completed / how many remain
      2) Elapsed time
      3) Estimated remaining time for the pipeline
    """
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

def get_unique_filename(folder_path, base_name, extension):
    """
    Generates a unique filename (e.g., base_name_1.extension, base_name_2.extension, etc.)
    inside the specified folder_path.
    """
    i = 1
    while True:
        filename = os.path.join(folder_path, f"{base_name}_{i}.{extension}")
        if not os.path.exists(filename):
            return filename
        i += 1

##############################
#  1) Read PDB Atoms and Coordinates
##############################
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
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"PDB file '{filename}' not found.")
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    atom_name = line[12:16].strip()
                    res_name  = line[17:20].strip()
                    chain     = line[21].strip()
                    res_num   = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception as e:
                    raise ValueError(f"Error parsing line: {line.strip()} : {e}")
                atoms.append({
                    'atom_name': atom_name,
                    'res_name':  res_name,
                    'chain':     chain,
                    'res_num':   res_num,
                    'x':         x,
                    'y':         y,
                    'z':         z
                })
    if not atoms:
        raise ValueError("No ATOM records found in the PDB file.")
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
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception as e:
                    raise ValueError(f"Error parsing coordinates in line: {line.strip()} : {e}")
                coords.append([x, y, z])
    if not coords:
        raise ValueError("No atomic coordinates found in the PDB file.")
    return np.array(coords)

##############################
#  2) Create a 3D Grid Around the Protein
##############################
def create_grid(coords, spacing=0.5):
    """
    Generates a 3D grid around the given coordinates, with a 5 Å border.
    """
    if coords.size == 0:
        raise ValueError("Empty coordinates array provided to create_grid.")
    x_min, y_min, z_min = coords.min(axis=0) - 5
    x_max, y_max, z_max = coords.max(axis=0) + 5
    grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:spacing,
                                      y_min:y_max:spacing,
                                      z_min:z_max:spacing]
    return np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

##############################
#  3) Apply Difference of Gaussians (DoG)
##############################
def apply_dog_filter(grid, protein_coords, sigma1=1.0, sigma2=2.0):
    """
    Applies a Difference of Gaussians (DoG) operation to detect potential pockets.
    This is not a Biopython predictor, just a mathematical approach.
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

##############################
#  4) Extract Grid Points Above a DoG Threshold
##############################
def extract_pocket_points(grid, dog_filtered, threshold_percentile=95):
    """
    Returns the grid points whose DoG value is above a certain percentile.
    """
    threshold = np.percentile(dog_filtered, threshold_percentile)
    return grid[dog_filtered > threshold]

##############################
#  5) Cluster Pockets with DBSCAN
##############################
def cluster_pockets(pocket_points, eps=0.8, min_samples=5):
    """
    Clusters 'pocket_points' using DBSCAN, returning a list of pocket dictionaries.
    Raises an error if clustering results in no significant pockets.
    """
    if len(pocket_points) < 2:
        raise ValueError("Not enough pocket points to perform clustering.")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
    labels = clustering.labels_
    clustered_pockets = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    MAX_CLUSTER_VOLUME = 3000.0  # Filter to discard overly large clusters

    for cluster_id in unique_labels:
        cluster_coords = pocket_points[labels == cluster_id]
        if len(cluster_coords) < 5:
            continue
        centroid = cluster_coords.mean(axis=0)
        try:
            hull = ConvexHull(cluster_coords)
        except Exception:
            continue
        volume, surface_area = hull.volume, hull.area
        dogsite_score = min((volume / (surface_area + 1e-6)) * 10, 1.0)
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
    if not clustered_pockets:
        raise ValueError("Clustering resulted in no significant pockets.")
    return clustered_pockets

##############################
#  6) Save Pocket Centroids in a PDB File
##############################
def save_pockets_as_pdb(pockets, output_filename):
    """
    Saves the predicted pockets in a PDB file with a descriptive header.
    Each pocket is stored as a HETATM line (coordinates = centroid).
    """
    try:
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
    except Exception as e:
        raise IOError(f"Error writing pockets PDB file: {e}")

##############################
#  6b) Find Residues in the Pocket (Distance-based)
##############################
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
    return sorted(residues_in_pocket, key=lambda x: (x[0], x[2]))

##############################
#  6c) Save Only Certain Residues in a New PDB
##############################
def save_residues_as_pdb(atom_list, residues, output_filename):
    """
    Saves a PDB file containing only atoms belonging to the given 'residues',
    where each residue is (chain, res_name, res_num).
    """
    residue_set = set(residues)
    try:
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
    except Exception as e:
        raise IOError(f"Error writing residues PDB file: {e}")

##############################
#  7) Visualization Scripts Generators
##############################
def generate_pymol_script(original_pdb_name, pockets_pdb_name, pocket_residues_names, output_script):
    """
    Creates a PyMOL script that:
      - Opens the original protein
      - Opens the pocket centroids (pockets_pdb_name) and displays them as a transparent surface
      - Opens each pocket residue file as a transparent surface
    """
    try:
        with open(output_script, 'w') as file:
            file.write(f"load {original_pdb_name}\n")
            file.write("show cartoon\n")
            file.write("color gray, all\n")
            file.write("zoom\n\n")
            file.write(f"load {pockets_pdb_name}\n")
            file.write("show surface, resn POC\n")
            file.write("color yellow, resn POC\n")
            file.write("set transparency, 0.5, resn POC\n\n")
            colors = ["red", "green", "blue", "magenta", "cyan", "orange"]
            for i, residue_pdb_name in enumerate(pocket_residues_names):
                color = colors[i % len(colors)]
                file.write(f"load {residue_pdb_name}\n")
                file.write("show surface\n")
                file.write(f"color {color}, (all)\n")
                file.write("set transparency, 0.5, (all)\n\n")
            file.write("zoom\n")
    except Exception as e:
        raise IOError(f"Error writing PyMOL script: {e}")

def generate_chimera_script(original_pdb_name, pockets_pdb_name, pocket_residues_names, output_script):
    """
    Creates a Chimera script that:
      - Opens the original protein as a surface (mesh)
      - Opens the pocket centroids as spheres
      - Opens each pocket residue file as a transparent surface
    """
    try:
        with open(output_script, 'w') as file:
            file.write(f"open {original_pdb_name}\n")
            file.write("surface #0\n")
            file.write("surfrepr mesh #0\n\n")
            file.write(f"open {pockets_pdb_name}\n")
            file.write("select #1 & :POC\n")
            file.write("rep sphere sel\n")
            file.write("color yellow sel\n")
            file.write("~select\n")
            file.write("focus\n\n")
            colors = ["red", "green", "blue", "magenta", "cyan", "orange"]
            for i, residue_pdb_name in enumerate(pocket_residues_names, start=2):
                color = colors[(i - 2) % len(colors)]
                file.write(f"open {residue_pdb_name}\n")
                file.write(f"surface #{i}\n")
                file.write(f"transparency 50 #{i}\n")
                file.write(f"color {color} #{i}\n\n")
            file.write("focus\n")
    except Exception as e:
        raise IOError(f"Error writing Chimera script: {e}")

##############################
#  8) BindingSitePredictor Class
##############################
class BindingSitePredictor:
    """
    A class for predicting binding sites from a protein PDB file.
    It creates a 3D grid, applies a Difference of Gaussians filter,
    clusters potential pockets, finds residues near each pocket, and
    generates output files (PDB, PyMOL script, Chimera script) in a folder named after the protein.
    """
    def __init__(self, pdb_file, grid_spacing=0.5, dog_threshold_percentile=95, eps=0.8, min_samples=5, residue_distance=5.0):
        if not os.path.isfile(pdb_file):
            raise FileNotFoundError(f"Input PDB file '{pdb_file}' not found.")
        self.pdb_file = pdb_file
        self.grid_spacing = grid_spacing
        self.dog_threshold_percentile = dog_threshold_percentile
        self.eps = eps
        self.min_samples = min_samples
        self.residue_distance = residue_distance
        
        self.protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self.output_folder = self.protein_name + "_results"
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Copy the original PDB file into the output folder
        self.local_pdb = os.path.join(self.output_folder, f"{self.protein_name}.pdb")
        try:
            shutil.copy(self.pdb_file, self.local_pdb)
        except Exception as e:
            raise IOError(f"Error copying original PDB file: {e}")
    
    def run(self):
        start_time = time.time()
        TOTAL_STEPS = 7
        current_step = 0
        
        # 1. Read protein atoms and coordinates
        protein_atoms = read_pdb_atoms(self.pdb_file)
        protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        
        # 2. Create a 3D grid around the protein
        grid_points = create_grid(protein_coords, spacing=self.grid_spacing)
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        
        # 3. Apply Difference of Gaussians (DoG) filter
        dog_filtered = apply_dog_filter(grid_points, protein_coords)
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        
        # 4. Extract pocket candidate points from the grid
        pocket_candidates = extract_pocket_points(grid_points, dog_filtered, threshold_percentile=self.dog_threshold_percentile)
        if pocket_candidates.size == 0:
            raise ValueError("No pocket candidate points found. Check the DoG threshold or input data.")
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        
        # 5. Cluster the pocket candidate points
        pocket_clusters = cluster_pockets(pocket_candidates, eps=self.eps, min_samples=self.min_samples)
        if len(pocket_clusters) == 0:
            raise ValueError("No significant pockets found after clustering.")
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        
        # 6. Generate unique filenames inside the output folder and save outputs
        pockets_pdb = get_unique_filename(self.output_folder, "predicted_pockets", "pdb")
        pymol_script = get_unique_filename(self.output_folder, "visualize_pockets", "pml")
        chimera_script = get_unique_filename(self.output_folder, "visualize_pockets", "cmd")
        
        # Save pocket centroids to a PDB file
        save_pockets_as_pdb(pocket_clusters, pockets_pdb)
        print(f"✅ Pocket centroids saved as {pockets_pdb}")
        
        # For each pocket, find and save residues (atoms within a distance of the pocket centroid)
        pocket_residues_list = []
        for i, pocket in enumerate(pocket_clusters, start=1):
            residues = find_residues_in_pocket(protein_atoms, pocket['centroid'], distance_threshold=self.residue_distance)
            pocket_residues_pdb = get_unique_filename(self.output_folder, f"pocket_{i}_residues", "pdb")
            save_residues_as_pdb(protein_atoms, residues, pocket_residues_pdb)
            pocket_residues_list.append(pocket_residues_pdb)
            print(f"   Pocket {i} residues saved in: {pocket_residues_pdb}")
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time)
        
        # 7. Generate visualization scripts (PyMOL and Chimera)
        # For visualization scripts, use only base filenames (assumes you 'cd' into the folder)
        local_pdb_name = os.path.basename(self.local_pdb)
        pockets_pdb_name = os.path.basename(pockets_pdb)
        pocket_residues_names = [os.path.basename(x) for x in pocket_residues_list]
        
        generate_pymol_script(local_pdb_name, pockets_pdb_name, pocket_residues_names, pymol_script)
        print(f"✅ PyMOL script saved as {pymol_script}")
        
        generate_chimera_script(local_pdb_name, pockets_pdb_name, pocket_residues_names, chimera_script)
        print(f"✅ Chimera script saved as {chimera_script}")
        
        print("\n📝 NOTE: Please 'cd' into the folder", self.output_folder,
              "before running the .cmd or .pml scripts to avoid file not found errors.")
        print("Processing complete.")

##############################
#  9) MAIN EXECUTION
##############################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 detect_binding_sites.py <protein_file.pdb>")
    input_pdb_filename = sys.argv[1]
    try:
        predictor = BindingSitePredictor(
            pdb_file=input_pdb_filename,
            grid_spacing=0.5,
            dog_threshold_percentile=95,
            eps=0.8,
            min_samples=5,
            residue_distance=5.0
        )
        predictor.run()
    except Exception as e:
        sys.exit(f"Error: {e}")
