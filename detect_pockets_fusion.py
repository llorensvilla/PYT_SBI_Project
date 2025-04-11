#!/usr/bin/env python3
import os
import sys
import time
import shutil
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

########################################################################
# 1) Main Class: GeoPredictor
########################################################################
class GeoPredictor:
    """
    GeoPredictor:
    Detects binding sites from a protein PDB file.

    Workflow (Numbered Steps):
      1. Read protein atoms and compute coordinates.
      2. Dynamically adjust the grid spacing based on protein size.
      3. Create a 3D grid around the protein.
      4. Apply the GeoPredictor filter (Difference-of-Gaussians) with sub-step progress.
      5. Extract grid points above a dynamic threshold.
      6. Cluster candidate points using DBSCAN with fixed parameters (eps = 0.8, min_samples = 5)
         and filter clusters based on a maximum volume (relative to the protein's bounding box).
      7. Evaluate binding site quality based on nearby residues and compute an interaction score.
      8. Combine the geometric score and the interaction score into a CompositeScore.
      9. Save the predicted pockets (with their scores) and generate visualization scripts.

    Note:
      - In the saved pockets PDB file, the Occupancy field is set to 1.00 as a placeholder.
      - The CompositeScore is calculated as:
            CompositeScore = (geometric_weight * geometric_score) + (interaction_weight * interaction_score)
    """

    def __init__(self, pdb_file, grid_spacing=0.5, dog_threshold_percentile=95,
                 eps=0.8, min_samples=5, residue_distance=5.0,
                 geometric_weight=0.5, interaction_weight=0.5,
                 interaction_weights=None):
        """
        (0.1) Initialization and file check.
        Checks if the input PDB file exists and sets initial parameters.
        Copies the original PDB file into an output folder named after the protein.
        """
        if not os.path.isfile(pdb_file):
            raise FileNotFoundError(f"Input PDB file '{pdb_file}' not found.")
        self.pdb_file = pdb_file
        self.grid_spacing = grid_spacing           # Initial value; will be updated dynamically.
        self.dog_threshold_percentile = dog_threshold_percentile  # Initial value; updated dynamically.
        self.eps = eps                             # Fixed DBSCAN eps value (0.8).
        self.min_samples = min_samples             # Fixed DBSCAN min_samples value (5).
        self.residue_distance = residue_distance
        self.geometric_weight = geometric_weight
        self.interaction_weight = interaction_weight
        # Default interaction weights if none provided.
        self.interaction_weights = interaction_weights or {
            "hbond": 0.25,
            "ionic": 0.25,
            "metal": 0.15,
            "hydrophobic": 0.20,
            "aromatic": 0.15
        }
        self.protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self.output_folder = self.protein_name + "_results"
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
        # Copy original PDB into output folder.
        self.local_pdb = os.path.join(self.output_folder, f"{self.protein_name}.pdb")
        try:
            shutil.copy(self.pdb_file, self.local_pdb)
        except Exception as e:
            raise IOError(f"Error copying original PDB file: {e}")

    def run(self):
        """
        Executes the complete GeoPredictor pipeline.

        Numbered Steps:
          1) Read protein atoms and compute coordinates.
          2) Dynamically adjust grid spacing and GeoPredictor threshold.
          3) Create a 3D grid.
          4) Apply the GeoPredictor filter with sub-step progress updates.
          5) Extract candidate pocket points using the threshold.
          6) Cluster these points with DBSCAN (fixed parameters).
          7) Evaluate the binding site quality and compute the CompositeScore.
          8) Save the pockets and generate visualization scripts.
        """
        start_time = time.time()
        TOTAL_STEPS = 7
        current_step = 0

        # === Step 1: Read Protein Atoms and Compute Coordinates ===
        step_name = "Reading Protein Atoms"
        protein_atoms = self.read_pdb_atoms(self.pdb_file)
        protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # === Step 2: Dynamic Adjustment of Grid Spacing and Threshold ===
        self.grid_spacing = self.auto_spacing(protein_coords)
        print(f"Using auto grid spacing: {self.grid_spacing:.2f} Å (based on protein size)")
        self.dog_threshold_percentile = self.auto_dog_threshold(len(protein_atoms))
        print(f"Using auto GeoPredictor threshold percentile: {self.dog_threshold_percentile:.1f} (based on {len(protein_atoms)} atoms)")

        # === Step 3: Create a 3D Grid ===
        step_name = "Creating 3D Grid"
        grid_points = self.create_grid(protein_coords, spacing=self.grid_spacing)
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # === Step 4: Apply GeoPredictor Filter with Sub-Steps ===
        step_name = "Applying GeoPredictor Filter"
        gp_filtered = self.apply_dog_filter(grid_points, protein_coords, sigma1=1.0, sigma2=2.0, substep_interval=50)
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # === Step 5: Extract Pocket Candidate Points ===
        step_name = "Extracting Pocket Points"
        pocket_candidates = self.extract_pocket_points(grid_points, gp_filtered, threshold_percentile=self.dog_threshold_percentile)
        if pocket_candidates.size == 0:
            raise ValueError("No pocket candidate points found. Check the GeoPredictor threshold or input data.")
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # === Step 6: Cluster Candidate Points using DBSCAN (Fixed Parameters) ===
        eps_val = 0.8
        min_samples_val = 5
        print(f"Using DBSCAN parameters: eps = {eps_val:.2f}, min_samples = {min_samples_val}")
        pocket_clusters = self.cluster_pockets(pocket_candidates, protein_coords, eps=eps_val, min_samples=min_samples_val)
        if len(pocket_clusters) == 0:
            raise ValueError("Clustering resulted in no significant pockets.")
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # === Step 7: Evaluate Binding Site Quality and Compute CompositeScore ===
        for pocket in pocket_clusters:
            interaction_score = self.evaluate_binding_site_score(protein_atoms, pocket, distance_threshold=self.residue_distance, interaction_weights=self.interaction_weights)
            composite_score = (self.geometric_weight * pocket['dogsite_score'] +
                               self.interaction_weight * interaction_score)
            pocket['interaction_score'] = interaction_score
            pocket['composite_score'] = composite_score

        # === Step 8: Save Predicted Pockets and Generate Visualization Scripts ===
        step_name = "Saving Pockets and Residues"
        pockets_pdb = self.get_unique_filename(self.output_folder, "predicted_pockets", "pdb")
        pymol_script = self.get_unique_filename(self.output_folder, "visualize_pockets", "pml")
        chimera_script = self.get_unique_filename(self.output_folder, "visualize_pockets", "cmd")
        self.save_pockets_as_pdb(pocket_clusters, pockets_pdb)
        print(f"✅ Pocket centroids saved as {pockets_pdb}")
        pocket_residues_list = []
        for i, pocket in enumerate(pocket_clusters, start=1):
            residues = self.find_residues_in_pocket(protein_atoms, pocket['centroid'], distance_threshold=self.residue_distance)
            pocket_residues_pdb = self.get_unique_filename(self.output_folder, f"pocket_{i}_residues", "pdb")
            self.save_residues_as_pdb(protein_atoms, residues, pocket_residues_pdb)
            pocket_residues_list.append(pocket_residues_pdb)
            print(f"   Pocket {i} residues saved in: {pocket_residues_pdb}")
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # === Step 9: Generate Visualization Scripts for PyMOL and Chimera ===
        step_name = "Generating Visualization Scripts"
        local_pdb_path = os.path.abspath(self.local_pdb)
        pockets_pdb_path = os.path.abspath(pockets_pdb)
        pocket_residues_paths = [os.path.abspath(x) for x in pocket_residues_list]
        self.generate_pymol_script(local_pdb_path, pockets_pdb_path, pocket_residues_paths, pymol_script)
        print(f"✅ PyMOL script saved as {pymol_script}")
        self.generate_chimera_script(local_pdb_path, pockets_pdb_path, pocket_residues_paths, chimera_script)
        print(f"✅ Chimera script saved as {chimera_script}")
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"\nProcessing complete. Total script time: {total_runtime:.1f}s")

    # --------------------------------------------------------------------
    # Helper Methods (Numbered for Clarity)
    # --------------------------------------------------------------------

    def print_progress(self, current_step, total_steps, start_time, step_name):
        """
        (1) Prints the current progress of the pipeline.
        Displays the current step, total steps, elapsed time, and estimated time remaining.
        """
        elapsed = time.time() - start_time
        steps_left = total_steps - current_step
        progress_fraction = current_step / total_steps if total_steps > 0 else 1
        estimated_total = elapsed / progress_fraction if progress_fraction > 0 else 0.0
        remaining = estimated_total - elapsed
        print(f"[Step {current_step}/{total_steps} - {step_name}] "
              f"({steps_left} steps left) "
              f"Elapsed: {elapsed:.1f}s | Estimated remaining: {remaining:.1f}s")

    def get_unique_filename(self, folder_path, base_name, extension):
        """
        (2) Generates a unique filename (e.g. base_name_1.extension, base_name_2.extension, etc.)
        in the specified folder.
        """
        i = 1
        while True:
            filename = os.path.join(folder_path, f"{base_name}_{i}.{extension}")
            if not os.path.exists(filename):
                return filename
            i += 1

    def auto_spacing(self, protein_coords):
        """
        (3) Dynamically calculates grid spacing based on protein size.
        - For proteins > 150 Å: returns 1.0 Å.
        - For proteins > 75 Å: returns 0.75 Å.
        - Otherwise: returns 0.5 Å.
        """
        protein_size = np.linalg.norm(protein_coords.max(axis=0) - protein_coords.min(axis=0))
        if protein_size > 150:
            return 1.0
        elif protein_size > 75:
            return 0.75
        else:
            return 0.5

    def auto_dog_threshold(self, num_atoms):
        """
        (4) Determines the GeoPredictor threshold percentile based on the number of atoms.
        - If num_atoms > 10000: returns 97.
        - If num_atoms < 1000: returns 90.
        - Otherwise: returns 95.
        """
        if num_atoms > 10000:
            return 97
        elif num_atoms < 1000:
            return 90
        else:
            return 95

    def read_pdb_atoms(self, filename):
        """
        (5) Reads the PDB file and returns a list of atom dictionaries.
        Each dictionary contains:
          - 'atom_name', 'res_name', 'chain', 'res_num', 'x', 'y', and 'z'
        Only lines starting with "ATOM" are processed.
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

    def create_grid(self, coords, spacing=0.5):
        """
        (6) Creates a 3D grid around the protein's coordinates with a 5 Å border.
        Returns the grid as an array of (x, y, z) points.
        """
        if coords.size == 0:
            raise ValueError("Empty coordinates array provided to create_grid.")
        x_min, y_min, z_min = coords.min(axis=0) - 5
        x_max, y_max, z_max = coords.max(axis=0) + 5
        grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:spacing,
                                          y_min:y_max:spacing,
                                          z_min:z_max:spacing]
        return np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    def apply_dog_filter(self, grid, protein_coords, sigma1=1.0, sigma2=2.0, substep_interval=50):
        """
        (7) Applies a Difference-of-Gaussians (GeoPredictor) filter to detect potential pockets.
        - Iterates over each protein coordinate.
        - Provides sub-step progress updates every 'substep_interval' iterations.
        - Returns normalized filter values for the entire grid.
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

    def extract_pocket_points(self, grid, dog_filtered, threshold_percentile=95):
        """
        (8) Extracts grid points whose GeoPredictor filter value is above a specified percentile.
        Returns the subset of grid points that are candidate pocket points.
        """
        threshold = np.percentile(dog_filtered, threshold_percentile)
        return grid[dog_filtered > threshold]

    def cluster_pockets(self, pocket_points, protein_coords, eps, min_samples):
        """
        (9) Clusters candidate pocket points using DBSCAN.
        - Filters clusters to keep only those with a volume less than 10% of the protein's bounding box volume.
        - Computes a geometric (GeoPredictor) score for each cluster.
        Returns a list of dictionaries for each clustered pocket.
        """
        if len(pocket_points) < 2:
            raise ValueError("Not enough pocket points to perform clustering.")
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
        labels = clustering.labels_
        clustered_pockets = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        bbox = protein_coords.max(axis=0) - protein_coords.min(axis=0)
        bbox_volume = np.prod(bbox)
        MAX_CLUSTER_VOLUME = bbox_volume * 0.1
        for cluster_id in unique_labels:
            cluster_coords = pocket_points[labels == cluster_id]
            if len(cluster_coords) < 5:
                continue
            try:
                hull = ConvexHull(cluster_coords)
            except Exception:
                continue
            volume, surface_area = hull.volume, hull.area
            geo_score = min((volume / (surface_area + 1e-6)) * 10, 1.0)
            if volume > MAX_CLUSTER_VOLUME:
                continue
            centroid = cluster_coords.mean(axis=0)
            clustered_pockets.append({
                'cluster_id': cluster_id,
                'coords': cluster_coords,
                'centroid': centroid,
                'volume': volume,
                'surface_area': surface_area,
                'dogsite_score': geo_score
            })
        if not clustered_pockets:
            raise ValueError("Clustering resulted in no significant pockets.")
        return clustered_pockets

    def evaluate_binding_site_score(self, atom_list, pocket, distance_threshold=5.0, interaction_weights=None):
        """
        (10) Calculates an interaction score based on nearby residues.
        - Considers hydrogen bonds, ionic interactions, metal coordination, hydrophobic and aromatic interactions.
        - Returns a normalized score between 0 and 1.
        """
        weights = interaction_weights or {"hbond": 0.25, "ionic": 0.25, "metal": 0.15, "hydrophobic": 0.20, "aromatic": 0.15}
        hbond_residues = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "LYS", "ARG"}
        ionic_pos_residues = {"LYS", "ARG", "HIS"}
        ionic_neg_residues = {"ASP", "GLU"}
        metal_binding_residues = {"HIS", "CYS", "ASP", "GLU"}
        hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"}
        aromatic_residues = {"PHE", "TYR", "TRP"}
        residues = self.find_residues_in_pocket(atom_list, pocket['centroid'], distance_threshold)
        if not residues:
            return 0.0
        counts = {"hbond": 0, "ionic": 0, "metal": 0, "hydrophobic": 0, "aromatic": 0}
        for chain, res_name, res_num in residues:
            res = res_name.upper()
            if res in hbond_residues:
                counts["hbond"] += 1
            if res in ionic_pos_residues or res in ionic_neg_residues:
                counts["ionic"] += 1
            if res in metal_binding_residues:
                counts["metal"] += 1
            if res in hydrophobic_residues:
                counts["hydrophobic"] += 1
            if res in aromatic_residues:
                counts["aromatic"] += 1
        weighted_sum = sum(weights[key] * counts[key] for key in counts)
        max_per_residue = sum(weights.values())
        normalized_score = weighted_sum / (len(residues) * max_per_residue)
        return min(max(normalized_score, 0.0), 1.0)

    def find_residues_in_pocket(self, atom_list, centroid, distance_threshold=5.0):
        """
        (11) Identifies residues within a specified distance (default 5.0 Å) from a pocket centroid.
        Returns a sorted list of (chain, res_name, res_num) tuples.
        """
        residues_in_pocket = set()
        for atom in atom_list:
            x, y, z = atom['x'], atom['y'], atom['z']
            if np.linalg.norm(np.array([x, y, z]) - centroid) <= distance_threshold:
                residues_in_pocket.add((atom['chain'], atom['res_name'], atom['res_num']))
        return sorted(residues_in_pocket, key=lambda x: (x[0], x[2]))

    def save_residues_as_pdb(self, atom_list, residues, output_filename):
        """
        (12) Saves a PDB file containing only the atoms belonging to the specified residues.
        The file includes REMARK lines listing each residue.
        """
        try:
            with open(output_filename, 'w') as out:
                out.write("REMARK  Residues forming the pocket\n")
                out.write("REMARK  CHAIN, RESNAME, RESNUM\n")
                for r in residues:
                    out.write(f"REMARK  {r[0]} {r[1]} {r[2]}\n")
                out.write("REMARK\n")
                atom_id = 1
                for atom in atom_list:
                    if (atom['chain'], atom['res_name'], atom['res_num']) in set(residues):
                        out.write(f"ATOM  {atom_id:5d} {atom['atom_name']:^4s}"
                                  f"{atom['res_name']:>3s} {atom['chain']}{atom['res_num']:4d}    "
                                  f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00\n")
                        atom_id += 1
        except Exception as e:
            raise IOError(f"Error writing residues PDB file: {e}")

    def save_pockets_as_pdb(self, pockets, output_filename):
        """
        (13) Saves the predicted pockets in a PDB file with a descriptive header.
        The header specifies each field with fixed spacing.
        Each pocket is written as a HETATM record with:
          - Occupancy set to 1.00 (placeholder)
          - CompositeScore showing the combined geometric and interaction score.
        After each pocket line, a REMARK line is added with detailed information about that pocket.
        """
        try:
            pockets_sorted = sorted(pockets, key=lambda x: x['surface_area'], reverse=True)
            with open(output_filename, 'w') as file:
                file.write("REMARK  Generated by GeoPredictor (Enhanced Binding Site Prediction with Dynamic Adjustments)\n")
                file.write("REMARK  Dynamic Parameters: auto grid spacing, auto GeoPredictor threshold, normalized cluster volume\n")
                file.write("REMARK  Detailed information about each predicted pocket is provided below the column headers.\n")
                # Define and write the header line with fixed width formatting.
                file.write(f"{'HETATM':>6} {'ID':>5} {'Residue':<6} {'Chain':>5} {'ResNum':>6} "
                           f"{'X':>8} {'Y':>8} {'Z':>8} {'Occupancy':>10} {'CompositeScore':>18} {'Volume':>10} {'SurfaceArea':>14}\n")
                for i, pocket in enumerate(pockets_sorted):
                    x, y, z = pocket['centroid']
                    volume = pocket['volume']
                    surface_area = pocket['surface_area']
                    composite_score = pocket.get('composite_score', pocket['dogsite_score'])
                    # Write the HETATM line with fixed field widths.
                    record_line = (f"{'HETATM':>6}"  # Record name
                                   f"{(i+1):5d}"     # Serial number
                                   f"  POC"          # Atom name (here 'POC' is used as a placeholder)
                                   f" A"             # Chain identifier (using A)
                                   f"{1:6d}"         # Residue number (using 1 as a placeholder)
                                   f"    {x:8.3f}{y:8.3f}{z:8.3f}"
                                   f"  {1.00:10.2f}"  # Occupancy fixed to 1.00
                                   f"  {composite_score:18.2f}"
                                   f"  V={volume:10.2f}"
                                   f" SA={surface_area:14.2f}\n")
                    file.write(record_line)
                    # Write an additional remark line with detailed pocket information.
                    file.write(f"REMARK  Pocket {i+1}: Cluster ID = {pocket['cluster_id']}, Points = {len(pocket['coords'])}\n")
        except Exception as e:
            raise IOError(f"Error writing pockets PDB file: {e}")

    def generate_pymol_script(self, original_pdb_path, pockets_pdb_path, pocket_residues_paths, output_script):
        """
        (14) Generates a PyMOL script for visualization:
           1) Sets the background to black.
           2) Loads the original protein and displays it in cartoon style with partial transparency.
           3) Overlays a wire mesh on the protein.
           4) Loads the pocket centroids and displays them as spheres.
           5) Loads each pocket residue file and displays it as a colored, semi-transparent surface.
           6) Uses absolute paths so that the script can be executed from any location.
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

    def generate_chimera_script(self, original_pdb_path, pockets_pdb_path, pocket_residues_paths, output_script):
        """
        (15) Generates a Chimera script for visualization:
           1) Sets the background to solid black.
           2) Opens the original protein and displays its surface as a mesh.
           3) Opens the pocket centroids and displays them as spheres.
           4) Opens each pocket residue file and displays them as semi-transparent surfaces.
           5) Uses absolute paths so that the script runs from any location.
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

########################################################################
# 2) Main Execution
########################################################################
def main():
    """
    Main function to instantiate and run the GeoPredictor.
    """
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 detect_pockets_fusion_sam.py <protein_file.pdb>")
    input_pdb_filename = sys.argv[1]
    try:
        predictor = GeoPredictor(
            pdb_file=input_pdb_filename,
            grid_spacing=0.5,            # Initial value; updated dynamically.
            dog_threshold_percentile=95, # Initial value; updated dynamically.
            eps=0.8,                     # Fixed DBSCAN eps value.
            min_samples=5,               # Fixed DBSCAN min_samples value.
            residue_distance=5.0,
            geometric_weight=0.5,        # Weight for geometric score.
            interaction_weight=0.5,      # Weight for interaction score.
            interaction_weights={
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

if __name__ == "__main__":
    main()
