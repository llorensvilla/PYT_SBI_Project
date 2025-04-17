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

    Workflow:
      1. Read atoms
      2. Auto-adjust grid & threshold
      3. Build 3D grid
      4. Apply Difference-of-Gaussians filter
      5. Extract candidate points
      6. Cluster with DBSCAN + filter by volume and composite score
      7. Save pockets and generate PyMOL/Chimera scripts
    """

    def __init__(self, pdb_file,
                 grid_spacing=0.5,
                 dog_threshold_percentile=95,
                 eps=0.8,
                 min_samples=5,
                 residue_distance=5.0,
                 geometric_weight=0.5,
                 interaction_weight=0.5,
                 interaction_weights=None):
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
            "hbond": 0.25, "ionic": 0.25, "metal": 0.15,
            "hydrophobic": 0.20, "aromatic": 0.15
        }

        self.protein_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self.output_folder = f"{self.protein_name}_results"
        os.makedirs(self.output_folder, exist_ok=True)

        # copy original into results
        self.local_pdb = os.path.join(self.output_folder, f"{self.protein_name}.pdb")
        try:
            shutil.copy(self.pdb_file, self.local_pdb)
        except Exception as e:
            raise IOError(f"Error copying PDB: {e}")

    def run(self):
        """
        Execute all steps and generate visualization scripts.
        """
        start = time.time()
        TOTAL_STEPS = 7
        step = 0

        # 1) Read atoms
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Reading atoms")
        atoms = self.read_pdb_atoms(self.pdb_file)
        coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])

        # 2) Auto-adjust grid & threshold
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Auto grid & threshold")
        self.grid_spacing = self.auto_spacing(coords)
        print(f"  Grid spacing: {self.grid_spacing:.2f} Å")
        self.dog_threshold_percentile = self.auto_dog_threshold(len(atoms))
        print(f"  DoG threshold percentile: {self.dog_threshold_percentile:.1f}")

        # 3) Build grid
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Building 3D grid")
        grid = self.create_grid(coords, spacing=self.grid_spacing)

        # 4) Apply DoG filter
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Applying DoG filter")
        dog_vals = self.apply_dog_filter(grid, coords,
                                         sigma1=1.0, sigma2=2.0,
                                         substep_interval=50)

        # 5) Extract candidates
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Extracting candidates")
        candidates = self.extract_pocket_points(grid, dog_vals,
                                                threshold_percentile=self.dog_threshold_percentile)
        if candidates.size == 0:
            raise ValueError("No candidate points found.")

        # 6) Cluster + filter
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Clustering & filtering")
        eps_val = 0.8
        min_samples_val = 5
        min_volume_val = 15.0
        min_composite_val = 0.55
        print(f"  DBSCAN eps={eps_val}, min_samples={min_samples_val}, "
              f"min_volume={min_volume_val}, min_composite={min_composite_val}")
        pockets = self.cluster_pockets(
            candidates, coords,
            eps=eps_val, min_samples=min_samples_val,
            min_volume=min_volume_val,
            min_composite_score=min_composite_val
        )

        # 7) Save and generate scripts
        step += 1
        self.print_progress(step, TOTAL_STEPS, start, "Saving & scripting")

        pockets_pdb = self.get_unique_filename(self.output_folder, "predicted_pockets", "pdb")
        pymol_pml  = self.get_unique_filename(self.output_folder, "visualize_pockets", "pml")
        chimera_cmd= self.get_unique_filename(self.output_folder, "visualize_pockets", "cmd")

        self.save_pockets_as_pdb(pockets, pockets_pdb)
        print(f"  Saved: {pockets_pdb}")

        res_files = []
        for idx, p in enumerate(pockets, 1):
            res = self.find_residues_in_pocket(atoms, p['centroid'], distance_threshold=self.residue_distance)
            res_pdb = self.get_unique_filename(self.output_folder, f"pocket_{idx}_residues", "pdb")
            self.save_residues_as_pdb(atoms, res, res_pdb)
            res_files.append(res_pdb)
            print(f"   Pocket {idx} residues: {res_pdb}")

        # absolute paths for scripts
        abs_prot = os.path.abspath(self.local_pdb)
        abs_pock = os.path.abspath(pockets_pdb)
        abs_res  = [os.path.abspath(x) for x in res_files]

        self.generate_pymol_script(abs_prot, abs_pock, abs_res, pymol_pml)
        print(f"  PyMOL script: {pymol_pml}")

        self.generate_chimera_script(abs_prot, abs_pock, abs_res, chimera_cmd)
        print(f"  Chimera script: {chimera_cmd}")

        print(f"\nDone in {time.time()-start:.1f}s")

    # -------------------------------------------------------------------
    def print_progress(self, cur, total, start, name):
        elapsed = time.time() - start
        frac = cur/total if total else 1
        rem = (elapsed/frac - elapsed) if frac>0 else 0
        print(f"[{cur}/{total}] {name} — Elapsed: {elapsed:.1f}s, Remain: {rem:.1f}s")

    def get_unique_filename(self, folder, base, ext):
        i = 1
        while True:
            name = os.path.join(folder, f"{base}_{i}.{ext}")
            if not os.path.exists(name):
                return name
            i += 1

    def auto_spacing(self, coords):
        size = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
        return 1.0 if size>150 else 0.75 if size>75 else 0.5

    def auto_dog_threshold(self, num_atoms):
        if num_atoms>10000: return 97
        if num_atoms<1000:  return 90
        return 95

    def read_pdb_atoms(self, fn):
        atoms = []
        with open(fn) as f:
            for L in f:
                if L.startswith("ATOM"):
                    atoms.append({
                        'atom_name': L[12:16].strip(),
                        'res_name':  L[17:20].strip(),
                        'chain':     L[21].strip(),
                        'res_num':   int(L[22:26]),
                        'x':         float(L[30:38]),
                        'y':         float(L[38:46]),
                        'z':         float(L[46:54]),
                    })
        if not atoms: raise ValueError("No ATOM lines.")
        return atoms

    def create_grid(self, coords, spacing=0.5):
        mins = coords.min(axis=0)-5
        maxs = coords.max(axis=0)+5
        gx,gy,gz = np.mgrid[mins[0]:maxs[0]:spacing,
                             mins[1]:maxs[1]:spacing,
                             mins[2]:maxs[2]:spacing]
        return np.vstack((gx.ravel(), gy.ravel(), gz.ravel())).T

    def apply_dog_filter(self, grid, coords, sigma1, sigma2, substep_interval):
        dens = np.zeros(len(grid))
        start = time.time()
        for i,c in enumerate(coords,1):
            d = np.linalg.norm(grid-c,axis=1)
            dens += np.exp(-d**2/(2*sigma1**2))
            if i%substep_interval==0 or i==len(coords):
                e = time.time()-start; frac=i/len(coords)
                r = (e/frac-e) if frac>0 else 0
                print(f"    sub-step {i}/{len(coords)} Elapsed:{e:.1f}s Rem:{r:.1f}s")
        b1 = gaussian_filter(dens, sigma=sigma1)
        b2 = gaussian_filter(dens, sigma=sigma2)
        dog= b1-b2
        return (dog-dog.min())/(dog.max()-dog.min())

    def extract_pocket_points(self, grid, dog_vals, threshold_percentile):
        thr = np.percentile(dog_vals, threshold_percentile)
        return grid[dog_vals>thr]

    def cluster_pockets(self, pocket_points, protein_coords,
                        eps, min_samples,
                        min_volume=15.0,
                        min_composite_score=0.55):
        if len(pocket_points)<2:
            raise ValueError("Too few points.")
        labels = DBSCAN(eps=eps,min_samples=min_samples).fit(pocket_points).labels_
        unique = set(labels)-{-1}
        bbox = protein_coords.max(axis=0)-protein_coords.min(axis=0)
        max_vol = np.prod(bbox)*0.1
        pockets=[]
        for cid in unique:
            pts = pocket_points[labels==cid]
            if len(pts)<5: continue
            try:
                hull=ConvexHull(pts)
            except: 
                continue
            vol,area = hull.volume, hull.area
            geo = min((vol/(area+1e-6))*10,1.0)
            if vol>max_vol or vol<min_volume: 
                continue
            center = pts.mean(axis=0)
            inter = self.evaluate_binding_site_score(
                atom_list=self.read_pdb_atoms(self.pdb_file),
                pocket={'centroid':center},
                distance_threshold=self.residue_distance,
                interaction_weights=self.interaction_weights
            )
            comp = (self.geometric_weight*geo + self.interaction_weight*inter)
            if comp<min_composite_score:
                continue
            pockets.append({
                'cluster_id':cid,
                'coords':pts,
                'centroid':center,
                'volume':vol,
                'surface_area':area,
                'dogsite_score':geo,
                'interaction_score':inter,
                'composite_score':comp
            })
        if not pockets:
            raise ValueError("No pockets after filtering.")
        return pockets

    def evaluate_binding_site_score(self, atom_list, pocket,
                                    distance_threshold, interaction_weights):
        weights = interaction_weights or self.interaction_weights
        hb = {"SER","THR","TYR","ASN","GLN","HIS","LYS","ARG"}
        io = {"ASP","GLU","LYS","ARG","HIS"}
        mt = {"HIS","CYS","ASP","GLU"}
        hp = {"ALA","VAL","LEU","ILE","MET","PHE","TRP"}
        ar = {"PHE","TYR","TRP"}

        res = self.find_residues_in_pocket(atom_list, pocket['centroid'], distance_threshold)
        if not res: return 0.0
        counts = {"hbond":0,"ionic":0,"metal":0,"hydrophobic":0,"aromatic":0}
        for _,rn,_ in res:
            r=rn.upper()
            if r in hb: counts["hbond"]+=1
            if r in io: counts["ionic"]+=1
            if r in mt: counts["metal"]+=1
            if r in hp: counts["hydrophobic"]+=1
            if r in ar: counts["aromatic"]+=1

        total = sum(weights.values())
        score = sum(weights[k]*counts[k] for k in counts)/(len(res)*total)
        return max(0.0,min(score,1.0))

    def find_residues_in_pocket(self, atom_list, centroid, distance_threshold):
        nearby=set()
        for a in atom_list:
            d=np.linalg.norm(np.array([a['x'],a['y'],a['z']])-centroid)
            if d<=distance_threshold:
                nearby.add((a['chain'],a['res_name'],a['res_num']))
        return sorted(nearby,key=lambda x:(x[0],x[2]))

    def save_residues_as_pdb(self, atom_list, residues, out_fn):
        try:
            with open(out_fn,'w') as out:
                out.write("REMARK  Residues forming the pocket\n")
                out.write("REMARK  CHAIN, RESNAME, RESNUM\n")
                for r in residues:
                    out.write(f"REMARK  {r[0]} {r[1]} {r[2]}\n")
                out.write("REMARK\n")
                atom_id=1
                for a in atom_list:
                    key=(a['chain'],a['res_name'],a['res_num'])
                    if key in set(residues):
                        out.write(f"ATOM  {atom_id:5d} {a['atom_name']:^4s}"
                                  f"{a['res_name']:>3s} {a['chain']}{a['res_num']:4d}    "
                                  f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00\n")
                        atom_id+=1
        except Exception as e:
            raise IOError(f"Error writing residues PDB: {e}")

    def save_pockets_as_pdb(self, pockets, out_fn):
        try:
            sorted_p = sorted(pockets, key=lambda x:x['surface_area'], reverse=True)
            with open(out_fn,'w') as f:
                f.write("REMARK  GeoPredictor pockets (filtered)\n")
                f.write("REMARK  Columns: HETATM,ID,X,Y,Z,Occ,CompScore,Vol,SurfArea\n")
                f.write(f"{'HETATM':>6} {'ID':>5} {'X':>8} {'Y':>8} {'Z':>8} "
                        f"{'Occupancy':>10} {'CompositeScore':>18} {'Volume':>10} {'SurfaceArea':>14}\n")
                for i,p in enumerate(sorted_p,1):
                    x,y,z=p['centroid']
                    f.write(f"{'HETATM':>6}{i:5d}"
                            f"{x:8.3f}{y:8.3f}{z:8.3f}"
                            f"{1.00:10.2f}{p['composite_score']:18.2f}"
                            f"{p['volume']:10.2f}{p['surface_area']:14.2f}\n")
                    f.write(f"REMARK  Pocket {i}: cluster={p['cluster_id']}, points={len(p['coords'])}\n")
        except Exception as e:
            raise IOError(f"Error writing pockets PDB: {e}")

    def generate_pymol_script(self, original_pdb, pockets_pdb, residues_pdbs, out_script):
        """
        (14) PyMOL script:
         - semi-transparent mesh on protein
         - spheres for centroids
         - colored transparent surfaces for pockets
        """
        try:
            with open(out_script,'w') as f:
                f.write("bg_color black\n")
                f.write(f"load {original_pdb}, protein\n")
                f.write("hide everything, protein\n\n")
                f.write("show cartoon, protein\n")
                f.write("color gray70, protein\n")
                f.write("set cartoon_transparency, 0.2, protein\n\n")
                f.write("show mesh, protein\n")
                f.write("color brown50, protein\n")
                f.write("set mesh_as_cylinders, 1\n")
                # make mesh semi-transparent so pockets show through
                f.write("set mesh_transparency, 0.7, protein\n\n")

                f.write(f"load {pockets_pdb}, pockets\n")
                f.write("hide everything, pockets\n")
                f.write("show spheres, pockets\n")
                f.write("set sphere_scale, 0.4, pockets\n\n")

                colors = ["yellow","red","green","blue","magenta","cyan","orange","purple","lime","teal"]
                for idx,res_pdb in enumerate(residues_pdbs,1):
                    obj = f"pocket_res_{idx}"
                    f.write(f"load {res_pdb}, {obj}\n")
                    f.write(f"hide everything, {obj}\n")
                    f.write(f"show surface, {obj}\n")
                    f.write(f"color {colors[(idx-1)%len(colors)]}, {obj}\n")
                    f.write("set transparency, 0.3, %s\n\n" % obj)

                f.write("zoom all\n")
        except Exception as e:
            raise IOError(f"Error writing PyMOL script: {e}")

    def generate_chimera_script(self, original_pdb, pockets_pdb, residues_pdbs, out_script):
        """
        (15) Chimera script:
         - semi-transparent protein mesh
         - spheres for centroids
         - colored transparent surfaces
        """
        try:
            with open(out_script,'w') as f:
                f.write("background solid black\n")
                f.write(f"open {original_pdb}\n")
                f.write("surface #0\n")
                f.write("surfrepr mesh #0\n")
                # make mesh 70% transparent
                f.write("transparency 70 #0\n\n")

                f.write(f"open {pockets_pdb}\n")
                f.write("select #1 & :POC\n")
                f.write("rep sphere sel\n")
                f.write("color yellow sel\n")
                f.write("~select\n")
                f.write("focus\n\n")

                colors = ["red","green","blue","magenta","cyan","orange","purple","lime"]
                for idx,res_pdb in enumerate(residues_pdbs, start=2):
                    f.write(f"open {res_pdb}\n")
                    f.write(f"surface #{idx}\n")
                    f.write(f"transparency 50 #{idx}\n")
                    f.write(f"color {colors[(idx-2)%len(colors)]} #{idx}\n\n")

                f.write("focus\n")
        except Exception as e:
            raise IOError(f"Error writing Chimera script: {e}")

########################################################################
# 2) Main Execution
########################################################################
def main():
    if len(sys.argv)<2:
        sys.exit("Usage: python3 detect_pockets_fusion_sam.py <protein_file.pdb>")
    predictor = GeoPredictor(pdb_file=sys.argv[1])
    predictor.run()

if __name__=="__main__":
    main()
