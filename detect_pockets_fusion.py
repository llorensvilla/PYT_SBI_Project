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
def print_progress(current_step, total_steps, start_time, step_name):
    """
    Prints a progress message showing:
      1) Current step vs. total steps.
      2) Step name.
      3) Elapsed time.
      4) Estimated remaining time for the entire pipeline.
    """
    elapsed = time.time() - start_time
    steps_left = total_steps - current_step
    progress_fraction = current_step / total_steps if total_steps > 0 else 1

    if progress_fraction > 0:
        estimated_total_time = elapsed / progress_fraction
    else:
        estimated_total_time = 0.0

    remaining = estimated_total_time - elapsed

    print(f"[Step {current_step}/{total_steps} - {step_name}] "
          f"({steps_left} steps left) "
          f"Elapsed: {elapsed:.1f}s | Estimated remaining: {remaining:.1f}s")


def get_unique_filename(folder_path, base_name, extension):
    """
    Generates a unique filename (base_name_1.extension, base_name_2.extension, etc.)
    within the specified folder_path.
    """
    i = 1
    while True:
        filename = os.path.join(folder_path, f"{base_name}_{i}.{extension}")
        if not os.path.exists(filename):
            return filename
        i += 1


##############################
#  New Dynamic Parameter Functions
##############################
def auto_spacing(protein_coords):
    """
    Determines the grid spacing dynamically based on the protein size.
    For large proteins: spacing = 1.0 Å,
    for intermediate proteins: spacing = 0.75 Å,
    otherwise: spacing = 0.5 Å.
    """
    protein_size = np.linalg.norm(protein_coords.max(axis=0) - protein_coords.min(axis=0))
    if protein_size > 150:
        return 1.0
    elif protein_size > 75:
        return 0.75
    else:
        return 0.5


def auto_dog_threshold(num_atoms):
    """
    Adjusts the GeoPredictor (DoG) threshold percentile based on the total number of atoms.
      - If num_atoms > 10000: percentile = 97
      - If num_atoms < 1000: percentile = 90
      - Otherwise: percentile = 95
    """
    if num_atoms > 10000:
        return 97
    elif num_atoms < 1000:
        return 90
    else:
        return 95


##############################
#  1) Read PDB Atoms / Coordinates
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
    Only lines starting with "ATOM" are considered.
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
    Extracts atomic coordinates (x, y, z) from a PDB file.
    Only lines starting with "ATOM" are processed.
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
#  2) Create a 3D Grid
##############################
def create_grid(coords, spacing=0.5):
    """
    Generates a 3D grid around the given coordinates, adding a 5 Å border.
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
#  3) Apply GeoPredictor Filter with Sub-Steps
##############################
def apply_dog_filter(grid, protein_coords, sigma1=1.0, sigma2=2.0, substep_interval=50):
    """
    Applies a Difference of Gaussians (DoG) operation (GeoPredictor filter) to detect potential pockets.
    Provides sub-step progress updates.
    
    substep_interval: number of protein coordinates processed before printing progress.
    """
    start_substep_time = time.time()
    density = np.zeros(len(grid))
    N = len(protein_coords)
    for i, coord in enumerate(protein_coords, start=1):
        dist = np.linalg.norm(grid - coord, axis=1)
        density += np.exp(-dist**2 / (2 * sigma1**2))
        if i % substep_interval == 0 or i == N:
            fraction_done = i / N
            elapsed_sub = time.time() - start_substep_time
            estimated_total = (elapsed_sub / fraction_done) if fraction_done > 0 else 0
            remain = estimated_total - elapsed_sub
            print(f"    [GeoPredictor sub-step] Processed {i}/{N} coords. "
                  f"Elapsed={elapsed_sub:.1f}s | Est. total={estimated_total:.1f}s | Remain={remain:.1f}s")
    blurred1 = gaussian_filter(density, sigma=sigma1)
    blurred2 = gaussian_filter(density, sigma=sigma2)
    dog_result = blurred1 - blurred2
    return (dog_result - np.min(dog_result)) / (np.max(dog_result) - np.min(dog_result))


##############################
#  4) Extract Points Above Threshold
##############################
def extract_pocket_points(grid, dog_filtered, threshold_percentile=95):
    """
    Returns the grid points whose GeoPredictor (DoG) value is above a given percentile threshold.
    """
    threshold = np.percentile(dog_filtered, threshold_percentile)
    return grid[dog_filtered > threshold]


###############################################################
#  5) Cluster Pockets with DBSCAN (Dynamic MAX_CLUSTER_VOLUME)
###############################################################
def cluster_pockets(pocket_points, protein_coords, eps, min_samples):
    """
    Clusters candidate pocket points using DBSCAN and returns a list of pocket dictionaries.
    
    Dynamic adjustments:
      - MAX_CLUSTER_VOLUME is set to 10% of the protein's bounding box volume.
      
    Each pocket dictionary contains:
      'cluster_id', 'coords', 'centroid', 'volume', 'surface_area', and 'dogsite_score' (the geometric score).
    """
    if len(pocket_points) < 2:
        raise ValueError("Not enough pocket points to perform clustering.")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
    labels = clustering.labels_
    clustered_pockets = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    # Compute maximum allowed volume based on the protein's bounding box
    bbox = protein_coords.max(axis=0) - protein_coords.min(axis=0)
    bbox_volume = np.prod(bbox)
    MAX_CLUSTER_VOLUME = bbox_volume * 0.1  # 10% of bounding box volume

    for cluster_id in unique_labels:
        cluster_coords = pocket_points[labels == cluster_id]
        if len(cluster_coords) < 5:
            continue
        try:
            hull = ConvexHull(cluster_coords)
        except Exception:
            continue
        volume, surface_area = hull.volume, hull.area
        gp_score = min((volume / (surface_area + 1e-6)) * 10, 1.0)
        if volume > MAX_CLUSTER_VOLUME:
            continue
        centroid = cluster_coords.mean(axis=0)
        clustered_pockets.append({
            'cluster_id': cluster_id,
            'coords': cluster_coords,
            'centroid': centroid,
            'volume': volume,
            'surface_area': surface_area,
            'dogsite_score': gp_score  # GeoPredictor geometric score.
        })

    if not clustered_pockets:
        raise ValueError("Clustering resulted in no significant pockets.")
    return clustered_pockets


##############################
#  Additional Helper: Evaluate Binding Site Quality
##############################
def evaluate_binding_site_score(atom_list, pocket, distance_threshold=5.0, interaction_weights=None):
    """
    Evaluates the binding site quality by analyzing residues within a given distance of the pocket centroid.
    Uses heuristics based on non-covalent interactions:
      - Hydrogen bonds, ionic interactions, metal coordination, hydrophobic and aromatic interactions.
    Calculates a weighted sum and normalizes it to a value between 0 and 1.
    """
    default_weights = {"hbond": 0.25, "ionic": 0.25, "metal": 0.15, "hydrophobic": 0.20, "aromatic": 0.15}
    if interaction_weights is None:
        interaction_weights = default_weights

    # Define residue groups for each type of interaction (using standard three-letter codes)
    hbond_residues = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "LYS", "ARG"}
    ionic_pos_residues = {"LYS", "ARG", "HIS"}
    ionic_neg_residues = {"ASP", "GLU"}
    metal_binding_residues = {"HIS", "CYS", "ASP", "GLU"}
    hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"}
    aromatic_residues = {"PHE", "TYR", "TRP"}

    residues = find_residues_in_pocket(atom_list, pocket['centroid'], distance_threshold)
    if not residues:
        return 0.0

    counts = {"hbond": 0, "ionic": 0, "metal": 0, "hydrophobic": 0, "aromatic": 0}
    for chain, res_name, res_num in residues:
        res_name = res_name.upper()
        if res_name in hbond_residues:
            counts["hbond"] += 1
        if res_name in ionic_pos_residues or res_name in ionic_neg_residues:
            counts["ionic"] += 1
        if res_name in metal_binding_residues:
            counts["metal"] += 1
        if res_name in hydrophobic_residues:
            counts["hydrophobic"] += 1
        if res_name in aromatic_residues:
            counts["aromatic"] += 1

    weighted_sum = sum(interaction_weights[key] * counts[key] for key in counts)
    max_per_residue = sum(interaction_weights.values())
    normalized_score = weighted_sum / (len(residues) * max_per_residue)
    return min(max(normalized_score, 0.0), 1.0)


##############################
#  6) Find and Save Pocket-Related Data
##############################
def find_residues_in_pocket(atom_list, centroid, distance_threshold=5.0):
    """
    Finds residues within a specified distance (default 5.0 Å) from the pocket centroid.
    Returns a sorted list of tuples: (chain, res_name, res_num).
    """
    residues_in_pocket = set()
    for atom in atom_list:
        x, y, z = atom['x'], atom['y'], atom['z']
        if np.linalg.norm(np.array([x, y, z]) - centroid) <= distance_threshold:
            residues_in_pocket.add((atom['chain'], atom['res_name'], atom['res_num']))
    return sorted(residues_in_pocket, key=lambda x: (x[0], x[2]))


def save_residues_as_pdb(atom_list, residues, output_filename):
    """
    Saves a PDB file containing only the atoms that belong to the given residues.
    Each residue is represented as a tuple: (chain, res_name, res_num).
    """
    residue_set = set(residues)
    try:
        with open(output_filename, 'w') as out:
            out.write("REMARK  Residues forming the pocket\n")
            out.write("REMARK  CHAIN, RESNAME, RESNUM\n")
            for r in residues:
                out.write(f"REMARK  {r[0]} {r[1]} {r[2]}\n")
            out.write("REMARK\n")
            atom_id = 1
            for atom in atom_list:
                if (atom['chain'], atom['res_name'], atom['res_num']) in residue_set:
                    out.write(
                        f"ATOM  {atom_id:5d} {atom['atom_name']:^4s}"
                        f"{atom['res_name']:>3s} {atom['chain']}{atom['res_num']:4d}    "
                        f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00\n"
                    )
                    atom_id += 1
    except Exception as e:
        raise IOError(f"Error writing residues PDB file: {e}")


def save_pockets_as_pdb(pockets, output_filename):
    """
    Saves the predicted pockets in a PDB file with a descriptive header.
    Each pocket is stored as a HETATM line (using the centroid coordinates) and
    sorted by surface area. The header indicates that dynamic parameters were used,
    and the composite score (combining geometric and interaction scores) is reported.
    """
    try:
        pockets_sorted = sorted(pockets, key=lambda x: x['surface_area'], reverse=True)
        with open(output_filename, 'w') as file:
            file.write("REMARK  Generated by GeoPredictor (Enhanced Binding Site Prediction with Dynamic Adjustments)\n")
            file.write("REMARK  Dynamic Parameters: auto grid spacing, auto GeoPredictor threshold, normalized cluster volume\n")
            file.write("REMARK  Columns:\n")
            file.write("HETATM  ID  Residue  Chain  ResNum  X      Y      Z      Occupancy  CompositeScore  Volume   SurfaceArea\n")
            for i, pocket in enumerate(pockets_sorted):
                x, y, z = pocket['centroid']
                volume = pocket['volume']
                surface_area = pocket['surface_area']
                composite_score = pocket.get('composite_score', pocket['dogsite_score'])
                file.write(
                    f"HETATM{(i+1):5d}  POC POC A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  {composite_score:5.2f}  "
                    f"V={volume:6.2f} SA={surface_area:6.2f}\n"
                )
    except Exception as e:
        raise IOError(f"Error writing pockets PDB file: {e}")


##############################
#  7) Visualization Scripts
##############################
def generate_pymol_script(original_pdb_path, pockets_pdb_path, pocket_residues_paths, output_script):
    """
    Generates a PyMOL script for visualization:
      - Displays the protein in cartoon style with partial transparency.
      - Overlays a wire mesh on the protein.
      - Shows pocket centroids as spheres.
      - Displays pocket residues as colored, semi-transparent surfaces.
      - Sets a black background.
    
    Absolute paths are used to ensure the script runs from any location.
    """
    try:
        with open(output_script, 'w') as file:
            file.write("bg_color black\n")
            file.write(f"load {original_pdb_path}, protein\n")
            file.write("hide everything, protein\n\n")
            file.write("show cartoon, protein\n")
            file.write("color gray70, protein\n")
            file.write("set cartoon_transparency, 0.2, protein\n\n")
            file.write("show mesh, protein\n")
            file.write("color brown50, protein\n")
            file.write("set mesh_as_cylinders, 1\n")
            file.write("set mesh_width, 0.6\n")
            file.write("set surface_quality, 2\n")
            file.write("set two_sided_lighting, on\n\n")
            file.write(f"load {pockets_pdb_path}, pockets\n")
            file.write("hide everything, pockets\n")
            file.write("show spheres, pockets\n")
            file.write("set sphere_scale, 0.4, pockets\n\n")
            colors = ["yellow", "red", "green", "blue", "magenta", "cyan", "orange", "purple", "lime", "teal"]
            for i in range(len(pocket_residues_paths)):
                color = colors[i % len(colors)]
                file.write(f"color {color}, pockets and id {i+1}\n")
            file.write("\n")
            for i, residue_pdb_path in enumerate(pocket_residues_paths):
                color = colors[i % len(colors)]
                object_name = f"residue_{i+1}"
                file.write(f"load {residue_pdb_path}, {object_name}\n")
                file.write(f"hide everything, {object_name}\n")
                file.write(f"show surface, {object_name}\n")
                file.write(f"color {color}, {object_name}\n")
                file.write(f"set transparency, 0.3, {object_name}\n\n")
            file.write("zoom all\n")
    except Exception as e:
        raise IOError(f"Error writing PyMOL script: {e}")


def generate_chimera_script(original_pdb_path, pockets_pdb_path, pocket_residues_paths, output_script):
    """
    Generates a Chimera script for visualization:
      - Opens the protein as a surface (mesh).
      - Opens pocket centroids as spheres.
      - Opens pocket residues as semi-transparent surfaces.
    
    Absolute paths are used to ensure the script runs from any location.
    """
    try:
        with open(output_script, 'w') as file:
            file.write("background solid black\n")
            file.write(f"open {original_pdb_path}\n")
            file.write("surface #0\n")
            file.write("surfrepr mesh #0\n\n")
            file.write(f"open {pockets_pdb_path}\n")
            file.write("select #1 & :POC\n")
            file.write("rep sphere sel\n")
            file.write("color yellow sel\n")
            file.write("~select\n")
            file.write("focus\n\n")
            colors = ["red", "green", "blue", "magenta", "cyan", "orange", "purple", "lime"]
            for i, residue_pdb_path in enumerate(pocket_residues_paths, start=2):
                color = colors[(i - 2) % len(colors)]
                file.write(f"open {residue_pdb_path}\n")
                file.write(f"surface #{i}\n")
                file.write(f"transparency 50 #{i}\n")
                file.write(f"color {color} #{i}\n\n")
            file.write("focus\n")
    except Exception as e:
        raise IOError(f"Error writing Chimera script: {e}")


##############################
#  8) GeoPredictor Class (Enhanced with Dynamic Adjustments)
##############################
class GeoPredictor:
    """
    GeoPredictor:
    Detects binding sites from a protein PDB file.
    The workflow consists of:
      1. Reading protein atoms and calculating coordinates.
      2. Dynamically adjusting the grid spacing based on protein size.
      3. Applying the GeoPredictor filter.
      4. Extracting grid points above a dynamically set threshold.
      5. Clustering candidate points using DBSCAN with fixed parameters (eps = 0.8, min_samples = 5)
         and normalizing the maximum cluster volume relative to the protein's bounding box.
      6. Evaluating each pocket using heuristics for non-covalent interactions and
         physico-chemical complementarity, then combining this score with the geometric score.
      7. Saving the results and generating visualization scripts for PyMOL and Chimera.
    
    Note: No external libraries for binding site prediction are used.
    """
    def __init__(self, pdb_file, grid_spacing=0.5, dog_threshold_percentile=95,
                 eps=0.8, min_samples=5, residue_distance=5.0,
                 geometric_weight=0.5, interaction_weight=0.5,
                 interaction_weights=None):
        if not os.path.isfile(pdb_file):
            raise FileNotFoundError(f"Input PDB file '{pdb_file}' not found.")
        self.pdb_file = pdb_file
        self.grid_spacing = grid_spacing           # Initial value; will be updated dynamically
        self.dog_threshold_percentile = dog_threshold_percentile  # Initial value; updated dynamically
        self.eps = eps                             # Fixed DBSCAN eps value
        self.min_samples = min_samples             # Fixed DBSCAN min_samples value
        self.residue_distance = residue_distance
        self.geometric_weight = geometric_weight
        self.interaction_weight = interaction_weight
        self.interaction_weights = interaction_weights  # Dictionary of weights for interactions

        self.protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self.output_folder = self.protein_name + "_results"
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        self.local_pdb = os.path.join(self.output_folder, f"{self.protein_name}.pdb")
        try:
            shutil.copy(self.pdb_file, self.local_pdb)
        except Exception as e:
            raise IOError(f"Error copying original PDB file: {e}")

    def run(self):
        start_time = time.time()
        TOTAL_STEPS = 7
        current_step = 0

        # Step 1: Read protein atoms and coordinates.
        step_name = "Reading Protein Atoms"
        protein_atoms = read_pdb_atoms(self.pdb_file)
        protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Dynamic adjustment: update grid spacing and GeoPredictor threshold.
        auto_spacing_val = auto_spacing(protein_coords)
        print(f"Using auto grid spacing: {auto_spacing_val:.2f} Å (based on protein size)")
        self.grid_spacing = auto_spacing_val

        auto_threshold = auto_dog_threshold(len(protein_atoms))
        print(f"Using auto GeoPredictor threshold percentile: {auto_threshold:.1f} (based on {len(protein_atoms)} atoms)")
        self.dog_threshold_percentile = auto_threshold

        # Step 2: Create grid with dynamic spacing.
        step_name = "Creating 3D Grid"
        grid_points = create_grid(protein_coords, spacing=self.grid_spacing)
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 3: Apply GeoPredictor filter with progress updates.
        step_name = "Applying GeoPredictor Filter"
        gp_filtered = apply_dog_filter(
            grid_points,
            protein_coords,
            sigma1=1.0,
            sigma2=2.0,
            substep_interval=50
        )
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 4: Extract pocket candidate points using the dynamic threshold.
        step_name = "Extracting Pocket Points"
        pocket_candidates = extract_pocket_points(grid_points, gp_filtered, threshold_percentile=self.dog_threshold_percentile)
        if pocket_candidates.size == 0:
            raise ValueError("No pocket candidate points found. Check the GeoPredictor threshold or input data.")
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Fixed DBSCAN parameters
        eps_val = 0.8
        min_samples_val = 5
        print(f"Using DBSCAN parameters: eps = {eps_val:.2f}, min_samples = {min_samples_val}")

        # Step 5: Cluster pocket candidate points with DBSCAN using fixed parameters.
        pocket_clusters = cluster_pockets(pocket_candidates, protein_coords, eps=eps_val, min_samples=min_samples_val)
        if len(pocket_clusters) == 0:
            raise ValueError("Clustering resulted in no significant pockets.")
        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 5b: Evaluate additional binding site quality (interaction score).
        for pocket in pocket_clusters:
            interaction_score = evaluate_binding_site_score(protein_atoms, pocket,
                                                              distance_threshold=self.residue_distance,
                                                              interaction_weights=self.interaction_weights)
            composite_score = (self.geometric_weight * pocket['dogsite_score'] +
                               self.interaction_weight * interaction_score)
            pocket['interaction_score'] = interaction_score
            pocket['composite_score'] = composite_score

        # Step 6: Save output PDB for pockets and associated residues.
        step_name = "Saving Pockets and Residues"
        pockets_pdb = get_unique_filename(self.output_folder, "predicted_pockets", "pdb")
        pymol_script = get_unique_filename(self.output_folder, "visualize_pockets", "pml")
        chimera_script = get_unique_filename(self.output_folder, "visualize_pockets", "cmd")

        save_pockets_as_pdb(pocket_clusters, pockets_pdb)
        print(f"✅ Pocket centroids saved as {pockets_pdb}")

        pocket_residues_list = []
        for i, pocket in enumerate(pocket_clusters, start=1):
            residues = find_residues_in_pocket(protein_atoms, pocket['centroid'], distance_threshold=self.residue_distance)
            pocket_residues_pdb = get_unique_filename(self.output_folder, f"pocket_{i}_residues", "pdb")
            save_residues_as_pdb(protein_atoms, residues, pocket_residues_pdb)
            pocket_residues_list.append(pocket_residues_pdb)
            print(f"   Pocket {i} residues saved in: {pocket_residues_pdb}")

        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 7: Generate visualization scripts for PyMOL and Chimera.
        step_name = "Generating Visualization Scripts"
        local_pdb_path = os.path.abspath(self.local_pdb)
        pockets_pdb_path = os.path.abspath(pockets_pdb)
        pocket_residues_paths = [os.path.abspath(x) for x in pocket_residues_list]

        generate_pymol_script(local_pdb_path, pockets_pdb_path, pocket_residues_paths, pymol_script)
        print(f"✅ PyMOL script saved as {pymol_script}")

        generate_chimera_script(local_pdb_path, pockets_pdb_path, pocket_residues_paths, chimera_script)
        print(f"✅ Chimera script saved as {chimera_script}")

        current_step += 1
        print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"\nProcessing complete. Total script time: {total_runtime:.1f}s")


##############################
#  9) Main Execution
##############################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 detect_pockets_fusion_sam.py <protein_file.pdb>")
    input_pdb_filename = sys.argv[1]
    try:
        predictor = GeoPredictor(
            pdb_file=input_pdb_filename,
            grid_spacing=0.5,          # Initial value; will be updated dynamically
            dog_threshold_percentile=95,  # Initial value; will be updated dynamically
            eps=0.8,                   # Fixed DBSCAN eps value
            min_samples=5,             # Fixed DBSCAN min_samples value
            residue_distance=5.0,
            geometric_weight=0.5,      # Weight for the geometric score
            interaction_weight=0.5,    # Weight for the interaction score
            interaction_weights={      # Weights for each type of non-covalent interaction
                "hbond": 0.25,
                "ionic": 0.25,
                "metal": 0.15,
                "hydrophobic": 0.20,
                "aromatic": 0.15
            }
        )
        predictor.run()
    except Exception as e:
        sys.exit(f"Error: {e}")
