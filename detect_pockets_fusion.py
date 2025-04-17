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
         and filter clusters based on a maximum volume (relative to the protein's bounding box)
         plus new minimum-volume and minimum-composite-score filters.
      7. Evaluate binding site quality based on nearby residues and compute an interaction score.
      8. Combine the geometric score and the interaction score into a CompositeScore.
      9. Save the predicted pockets (with their scores) and generate visualization scripts.

    Note:
      - In the saved pockets PDB file, the Occupancy field is set to 1.00 as a placeholder.
      - The CompositeScore is calculated as:
            CompositeScore = (geometric_weight * geometric_score)
                            + (interaction_weight * interaction_score)
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
        self.grid_spacing = grid_spacing
        self.dog_threshold_percentile = dog_threshold_percentile
        self.eps = eps
        self.min_samples = min_samples
        self.residue_distance = residue_distance
        self.geometric_weight = geometric_weight
        self.interaction_weight = interaction_weight
        self.interaction_weights = interaction_weights or {
            "hbond": 0.25,
            "ionic": 0.25,
            "metal": 0.15,
            "hydrophobic": 0.20,
            "aromatic": 0.15
        }
        self.protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self.output_folder = self.protein_name + "_results"
        os.makedirs(self.output_folder, exist_ok=True)

        # Copy original PDB into output folder.
        self.local_pdb = os.path.join(self.output_folder, f"{self.protein_name}.pdb")
        try:
            shutil.copy(self.pdb_file, self.local_pdb)
        except Exception as e:
            raise IOError(f"Error copying original PDB file: {e}")

    def run(self):
        """
        Executes the complete GeoPredictor pipeline.
        """
        start_time = time.time()
        TOTAL_STEPS = 7
        current_step = 0

        # Step 1: Read Protein Atoms and Compute Coordinates
        step_name = "Reading Protein Atoms"
        protein_atoms = self.read_pdb_atoms(self.pdb_file)
        protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 2: Dynamic Adjustment of Grid Spacing and Threshold
        self.grid_spacing = self.auto_spacing(protein_coords)
        print(f"Using auto grid spacing: {self.grid_spacing:.2f} Å")
        self.dog_threshold_percentile = self.auto_dog_threshold(len(protein_atoms))
        print(f"Using auto GeoPredictor threshold percentile: {self.dog_threshold_percentile:.1f}")

        # Step 3: Create a 3D Grid
        step_name = "Creating 3D Grid"
        grid_points = self.create_grid(protein_coords, spacing=self.grid_spacing)
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 4: Apply GeoPredictor Filter with Sub-Steps
        step_name = "Applying GeoPredictor Filter"
        gp_filtered = self.apply_dog_filter(
            grid_points, protein_coords,
            sigma1=1.0, sigma2=2.0,
            substep_interval=50
        )
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 5: Extract Pocket Candidate Points
        step_name = "Extracting Pocket Points"
        pocket_candidates = self.extract_pocket_points(
            grid_points, gp_filtered,
            threshold_percentile=self.dog_threshold_percentile
        )
        if pocket_candidates.size == 0:
            raise ValueError("No pocket candidate points found.")
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 6: Cluster Candidate Points using DBSCAN (with new filters)
        eps_val = 0.8
        min_samples_val = 5
        min_volume_val = 10.0
        min_composite_val = 0.60
        print(f"DBSCAN eps={eps_val}, min_samples={min_samples_val}, "
              f"min_volume={min_volume_val}, min_composite={min_composite_val}")

        try:
            pocket_clusters = self.cluster_pockets(
                pocket_candidates, protein_coords,
                eps=eps_val, min_samples=min_samples_val,
                min_volume=min_volume_val,
                min_composite_score=min_composite_val
            )
        except ValueError:
            # Fallback if all pockets get filtered out
            print("⚠️ No pockets passed strict filters – falling back to default clustering.")
            pocket_clusters = self.cluster_pockets(
                pocket_candidates, protein_coords,
                eps=eps_val, min_samples=min_samples_val,
                min_volume=0.0,
                min_composite_score=0.0
            )

        if not pocket_clusters:
            raise ValueError("Clustering resulted in no significant pockets.")
        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 7: Evaluate Binding Site Quality and Compute CompositeScore
        for pocket in pocket_clusters:
            interaction_score = self.evaluate_binding_site_score(
                protein_atoms, pocket,
                distance_threshold=self.residue_distance,
                interaction_weights=self.interaction_weights
            )
            composite_score = (
                self.geometric_weight * pocket['dogsite_score'] +
                self.interaction_weight * interaction_score
            )
            pocket['interaction_score'] = interaction_score
            pocket['composite_score'] = composite_score

        # Step 8: Save Predicted Pockets and Generate Visualization Scripts
        step_name = "Saving Pockets and Residues"
        pockets_pdb = self.get_unique_filename(self.output_folder, "predicted_pockets", "pdb")
        pymol_script = self.get_unique_filename(self.output_folder, "visualize_pockets", "pml")
        chimera_script = self.get_unique_filename(self.output_folder, "visualize_pockets", "cmd")

        self.save_pockets_as_pdb(pocket_clusters, pockets_pdb)
        print(f"✅ Pocket centroids saved as {pockets_pdb}")

        pocket_residues_list = []
        for i, pocket in enumerate(pocket_clusters, start=1):
            residues = self.find_residues_in_pocket(
                protein_atoms, pocket['centroid'],
                distance_threshold=self.residue_distance
            )
            pocket_residues_pdb = self.get_unique_filename(
                self.output_folder, f"pocket_{i}_residues", "pdb"
            )
            self.save_residues_as_pdb(protein_atoms, residues, pocket_residues_pdb)
            pocket_residues_list.append(pocket_residues_pdb)
            print(f"   Pocket {i} residues saved in: {pocket_residues_pdb}")

        current_step += 1
        self.print_progress(current_step, TOTAL_STEPS, start_time, step_name)

        # Step 9: Generate Visualization Scripts for PyMOL and Chimera
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
        print(f"\nProcessing complete. Total runtime: {end_time - start_time:.1f}s")

    # --------------------------------------------------------------------
    # Helper Methods (Numbered for Clarity)
    # --------------------------------------------------------------------

    def print_progress(self, current_step, total_steps, start_time, step_name):
        """
        (1) Prints the current progress of the pipeline.
        """
        elapsed = time.time() - start_time
        fraction = current_step / total_steps if total_steps > 0 else 1
        remaining = (elapsed / fraction - elapsed) if fraction > 0 else 0.0
        print(f"[Step {current_step}/{total_steps} - {step_name}] "
              f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")

    def get_unique_filename(self, folder_path, base_name, extension):
        """
        (2) Generates a unique filename in the specified folder.
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
        """
        size = np.linalg.norm(protein_coords.max(axis=0) - protein_coords.min(axis=0))
        if size > 150:
            return 1.0
        elif size > 75:
            return 0.75
        else:
            return 0.5

    def auto_dog_threshold(self, num_atoms):
        """
        (4) Determines the GeoPredictor threshold percentile based on atom count.
        """
        if num_atoms > 10000:
            return 97
        elif num_atoms < 1000:
            return 90
        else:
            return 95

    def read_pdb_atoms(self, filename):
        """
        (5) Reads the PDB file and returns a list of atom dicts.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"PDB file '{filename}' not found.")
        atoms = []
        with open(filename) as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    res_name  = line[17:20].strip()
                    chain     = line[21].strip()
                    res_num   = int(line[22:26])
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
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
            raise ValueError("No ATOM records found in the PDB.")
        return atoms

    def create_grid(self, coords, spacing=0.5):
        """
        (6) Creates a 3D grid around the protein with a 5 Å border.
        """
        mins = coords.min(axis=0) - 5
        maxs = coords.max(axis=0) + 5
        grid_x, grid_y, grid_z = np.mgrid[
            mins[0]:maxs[0]:spacing,
            mins[1]:maxs[1]:spacing,
            mins[2]:maxs[2]:spacing
        ]
        return np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    def apply_dog_filter(self, grid, protein_coords, sigma1=1.0, sigma2=2.0, substep_interval=50):
        """
        (7) Applies a Difference-of-Gaussians filter.
        """
        start = time.time()
        density = np.zeros(len(grid))
        N = len(protein_coords)
        for i, coord in enumerate(protein_coords, start=1):
            dist = np.linalg.norm(grid - coord, axis=1)
            density += np.exp(-dist**2 / (2 * sigma1**2))
            if i % substep_interval == 0 or i == N:
                frac = i / N
                elapsed = time.time() - start
                est_total = elapsed / frac if frac > 0 else elapsed
                print(f"    [GeoPredictor sub-step] {i}/{N} coords, elapsed={elapsed:.1f}s, est={est_total:.1f}s")
        b1 = gaussian_filter(density, sigma=sigma1)
        b2 = gaussian_filter(density, sigma=sigma2)
        result = b1 - b2
        return (result - result.min()) / (result.max() - result.min())

    def extract_pocket_points(self, grid, dog_filtered, threshold_percentile=95):
        """
        (8) Extracts grid points above a given percentile of DoG response.
        """
        thresh = np.percentile(dog_filtered, threshold_percentile)
        return grid[dog_filtered > thresh]

    def cluster_pockets(self, pocket_points, protein_coords,
                        eps, min_samples,
                        min_volume=15.0,
                        min_composite_score=0.55):
        """
        (9) Clusters points with DBSCAN and filters by volume and composite score.
        """
        if len(pocket_points) < 2:
            raise ValueError("Not enough points to cluster.")

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
        labels = clustering.labels_
        unique = set(labels) - {-1}

        # bounding‑box volume (10% max)
        bbox = protein_coords.max(axis=0) - protein_coords.min(axis=0)
        max_vol = np.prod(bbox) * 0.1

        pockets = []
        for cid in unique:
            pts = pocket_points[labels == cid]
            if len(pts) < 5:
                continue
            try:
                hull = ConvexHull(pts)
            except:
                continue
            vol, sa = hull.volume, hull.area
            geo_score = min((vol/(sa+1e-6))*10, 1.0)
            if vol > max_vol or vol < min_volume:
                continue

            centroid = pts.mean(axis=0)
            pocket = {
                'cluster_id': cid,
                'coords': pts,
                'centroid': centroid,
                'volume': vol,
                'surface_area': sa,
                'dogsite_score': geo_score
            }

            # compute interaction and composite
            inter = self.evaluate_binding_site_score(
                self.read_pdb_atoms(self.pdb_file),
                pocket,
                distance_threshold=self.residue_distance,
                interaction_weights=self.interaction_weights
            )
            comp = self.geometric_weight*geo_score + self.interaction_weight*inter
            if comp < min_composite_score:
                continue

            pocket['interaction_score'] = inter
            pocket['composite_score']  = comp
            pockets.append(pocket)

        if not pockets:
            raise ValueError("No significant pockets after filtering.")
        return pockets

    def evaluate_binding_site_score(self, atom_list, pocket, distance_threshold=5.0, interaction_weights=None):
        """
        (10) Calculates an interaction score between 0 and 1.
        """
        weights = interaction_weights or self.interaction_weights
        # residue sets...
        hbond = {"SER","THR","TYR","ASN","GLN","HIS","LYS","ARG"}
        ionic_pos = {"LYS","ARG","HIS"}
        ionic_neg = {"ASP","GLU"}
        metal = {"HIS","CYS","ASP","GLU"}
        hydro = {"ALA","VAL","LEU","ILE","MET","PHE","TRP"}
        arom  = {"PHE","TYR","TRP"}

        residues = self.find_residues_in_pocket(atom_list, pocket['centroid'], distance_threshold)
        if not residues:
            return 0.0

        counts = {"hbond":0, "ionic":0, "metal":0, "hydrophobic":0, "aromatic":0}
        for chain, name, num in residues:
            rn = name.upper()
            if rn in hbond:          counts["hbond"] += 1
            if rn in ionic_pos|ionic_neg: counts["ionic"] += 1
            if rn in metal:          counts["metal"] += 1
            if rn in hydro:          counts["hydrophobic"] += 1
            if rn in arom:           counts["aromatic"] += 1

        weighted = sum(weights[k]*counts[k] for k in counts)
        max_per = sum(weights.values())
        return min(max(weighted/(len(residues)*max_per), 0.0), 1.0)

    def find_residues_in_pocket(self, atom_list, centroid, distance_threshold=5.0):
        """
        (11) Finds residues within distance_threshold of the centroid.
        """
        found = set()
        c = np.array(centroid)
        for atom in atom_list:
            pos = np.array([atom['x'], atom['y'], atom['z']])
            if np.linalg.norm(pos - c) <= distance_threshold:
                found.add((atom['chain'], atom['res_name'], atom['res_num']))
        return sorted(found, key=lambda x: (x[0], x[2]))

    def save_residues_as_pdb(self, atom_list, residues, output_filename):
        """
        (12) Writes a PDB containing only the specified residues.
        """
        try:
            with open(output_filename, 'w') as out:
                out.write("REMARK  Residues forming the pocket\n")
                out.write("REMARK  CHAIN, RESNAME, RESNUM\n")
                for r in residues:
                    out.write(f"REMARK  {r[0]} {r[1]} {r[2]}\n")
                out.write("REMARK\n")
                atom_id = 1
                resset = set(residues)
                for atom in atom_list:
                    key = (atom['chain'], atom['res_name'], atom['res_num'])
                    if key in resset:
                        out.write(
                            f"ATOM  {atom_id:5d} {atom['atom_name']:^4s}"
                            f"{atom['res_name']:>3s} {atom['chain']}"
                            f"{atom['res_num']:4d}    "
                            f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                            f"  1.00  0.00\n"
                        )
                        atom_id += 1
        except Exception as e:
            raise IOError(f"Error writing residues PDB file: {e}")

    def save_pockets_as_pdb(self, pockets, output_filename):
        """
        (13) Saves pocket centroids with CompositeScore in a PDB.
        """
        try:
            sorted_p = sorted(pockets, key=lambda x: x['surface_area'], reverse=True)
            with open(output_filename, 'w') as f:
                f.write("REMARK  Generated by GeoPredictor\n")
                f.write("REMARK  Auto grid spacing, auto threshold, filtered pockets\n")
                f.write(
                    f"{'HETATM':>6} {'ID':>5} {'Residue':<6} {'Chain':>5} {'ResNum':>6} "
                    f"{'X':>8} {'Y':>8} {'Z':>8} {'Occupancy':>10} {'CompositeScore':>18} "
                    f"{'Volume':>10} {'SurfaceArea':>14}\n"
                )
                for i, p in enumerate(sorted_p, start=1):
                    x, y, z = p['centroid']
                    vol = p['volume']
                    sa  = p['surface_area']
                    comp = p['composite_score']
                    f.write(
                        f"{'HETATM':>6}{i:5d}  POC A{1:6d}"
                        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00"
                        f"{comp:18.2f}  V={vol:10.2f} SA={sa:14.2f}\n"
                    )
                    f.write(f"REMARK  Pocket {i}: Cluster ID={p['cluster_id']}, Points={len(p['coords'])}\n")
        except Exception as e:
            raise IOError(f"Error writing pockets PDB file: {e}")

    def generate_pymol_script(self, original_pdb_path, pockets_pdb_path,
                              pocket_residues_paths, output_script):
        """
        (14) Generates a PyMOL script for visualization.
        """
        try:
            with open(output_script, 'w') as f:
                f.write("bg_color black\n")
                f.write(f"load {original_pdb_path}, protein\n")
                f.write("hide everything, protein\n\n")
                f.write("show cartoon, protein\n")
                f.write("color gray70, protein\n")
                f.write("set cartoon_transparency, 0.2, protein\n\n")
                f.write("show mesh, protein\n")
                f.write("color brown50, protein\n")
                f.write("set mesh_as_cylinders, 1\n")
                f.write("set mesh_width, 0.6\n")
                f.write("set surface_quality, 2\n")
                f.write("set two_sided_lighting, on\n\n")
                f.write(f"load {pockets_pdb_path}, pockets\n")
                f.write("hide everything, pockets\n")
                f.write("show spheres, pockets\n")
                f.write("set sphere_scale, 0.4, pockets\n\n")
                colors = ["yellow","red","green","blue","magenta","cyan","orange","purple","lime","teal"]
                for i in range(len(pocket_residues_paths)):
                    color = colors[i % len(colors)]
                    f.write(f"color {color}, pockets and id {i+1}\n")
                f.write("\n")
                for i, path in enumerate(pocket_residues_paths):
                    color = colors[i % len(colors)]
                    obj = f"residue_{i+1}"
                    f.write(f"load {path}, {obj}\n")
                    f.write(f"hide everything, {obj}\n")
                    f.write(f"show surface, {obj}\n")
                    f.write(f"color {color}, {obj}\n")
                    f.write(f"set transparency, 0.3, {obj}\n\n")
                f.write("zoom all\n")
        except Exception as e:
            raise IOError(f"Error writing PyMOL script: {e}")

    def generate_chimera_script(self, original_pdb_path, pockets_pdb_path,
                                pocket_residues_paths, output_script):
        """
        (15) Generates a Chimera script for visualization.
        """
        try:
            with open(output_script, 'w') as f:
                f.write("background solid black\n")
                f.write(f"open {original_pdb_path}\n")
                f.write("surface #0\n")
                f.write("surfrepr mesh #0\n\n")
                f.write(f"open {pockets_pdb_path}\n")
                f.write("select #1 & :POC\n")
                f.write("rep sphere sel\n")
                f.write("color yellow sel\n")
                f.write("~select\n")
                f.write("focus\n\n")
                colors = ["red","green","blue","magenta","cyan","orange","purple","lime"]
                for i, path in enumerate(pocket_residues_paths, start=2):
                    color = colors[(i-2) % len(colors)]
                    f.write(f"open {path}\n")
                    f.write(f"surface #{i}\n")
                    f.write(f"transparency 50 #{i}\n")
                    f.write(f"color {color} #{i}\n\n")
                f.write("focus\n")
        except Exception as e:
            raise IOError(f"Error writing Chimera script: {e}")

########################################################################
# 2) Main Execution
########################################################################
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 detect_pockets_fusion_sam.py <protein_file.pdb>")
    input_pdb = sys.argv[1]
    try:
        predictor = GeoPredictor(
            pdb_file=input_pdb,
            grid_spacing=0.5,
            dog_threshold_percentile=95,
            eps=0.8,
            min_samples=5,
            residue_distance=5.0,
            geometric_weight=0.5,
            interaction_weight=0.5,
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
