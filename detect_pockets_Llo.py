import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

""" Function to generate unique filenames with increasing numbers """
def get_unique_filename(base_name, extension):
    """Genera un nombre de archivo único numerando secuencialmente"""
    i = 1
    while os.path.exists(f"{base_name}_{i}.{extension}"):
        i += 1
    return f"{base_name}_{i}.{extension}"

# ------------------------------------------------------------------------------
# 1) Parsear PDB
# ------------------------------------------------------------------------------

def read_pdb_atoms(filename):
    """
    Lee un fichero PDB y devuelve una lista de diccionarios con la información:
    {
        'atom_name': str,
        'res_name': str,
        'chain': str,
        'res_num': int,
        'x': float,
        'y': float,
        'z': float
    }
    Considera únicamente líneas que empiezan por 'ATOM' (o 'HETATM' si lo deseas).
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
    Extract atomic coordinates from a PDB file (sólo x, y, z).
    Sólo considera líneas ATOM, ignorando el resto.
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
# 2) Crear la malla 3D (grid)
# ------------------------------------------------------------------------------

def create_grid(coords, spacing=0.5):
    """
    Genera una malla 3D alrededor de las coordenadas dadas,
    con un borde de 5 Å alrededor (min y max).
    """
    x_min, y_min, z_min = coords.min(axis=0) - 5
    x_max, y_max, z_max = coords.max(axis=0) + 5
    grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:spacing,
                                      y_min:y_max:spacing,
                                      z_min:z_max:spacing]
    return np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

# ------------------------------------------------------------------------------
# 3) Aplicar Difference of Gaussians (DoG)
# ------------------------------------------------------------------------------

def apply_dog_filter(grid, protein_coords, sigma1=1.0, sigma2=2.0):
    """
    Aplica la operación Difference of Gaussians para detectar pockets.
    """
    density = np.zeros(len(grid))
    for coord in protein_coords:
        dist = np.linalg.norm(grid - coord, axis=1)
        density += np.exp(-dist**2 / (2 * sigma1**2))

    blurred1 = gaussian_filter(density, sigma=sigma1)
    blurred2 = gaussian_filter(density, sigma=sigma2)
    dog_result = blurred1 - blurred2

    # Normalizar entre 0 y 1 para facilidad de uso
    return (dog_result - np.min(dog_result)) / (np.max(dog_result) - np.min(dog_result))

# ------------------------------------------------------------------------------
# 4) Extraer puntos con alto valor de DoG
# ------------------------------------------------------------------------------

def extract_pocket_points(grid, dog_filtered, threshold_percentile=95):
    """
    Devuelve los puntos (del grid) que tienen un valor de DoG
    por encima de cierto percentil.
    """
    threshold = np.percentile(dog_filtered, threshold_percentile)
    return grid[dog_filtered > threshold]

# ------------------------------------------------------------------------------
# 5) Clusterizar pockets con DBSCAN
# ------------------------------------------------------------------------------

def cluster_pockets(pocket_points, eps=1.5, min_samples=5):
    """
    Agrupa los puntos 'pocket_points' en clusters usando DBSCAN.
    Retorna una lista de diccionarios con info de cada pocket.
    """
    if len(pocket_points) < 2:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_points)
    labels = clustering.labels_

    clustered_pockets = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Ruido

    # Ajusta el umbral de volumen que quieras (para descartar pockets gigantes)
    MAX_CLUSTER_VOLUME = 3000.0

    for cluster_id in unique_labels:
        cluster_coords = pocket_points[labels == cluster_id]
        if len(cluster_coords) < 5:
            continue

        centroid = cluster_coords.mean(axis=0)
        hull = ConvexHull(cluster_coords)
        volume, surface_area = hull.volume, hull.area
        dogsite_score = min((volume / (surface_area + 1e-6)) * 10, 1.0)

        # Filtro: descartar pockets con volumen excesivo
        if volume > MAX_CLUSTER_VOLUME:
            continue

        # Guardamos el pocket
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
# 6) Guardar pockets en un PDB con las coordenadas de sus CENTROIDES
# ------------------------------------------------------------------------------

def save_pockets_as_pdb(pockets, output_filename):
    """
    Guarda los pockets predichos en un archivo PDB con un header descriptivo
    + una HETATM por cada pocket (coordenadas = centroides).
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
# 6b) Encontrar residuos en el pocket (por distancia al centróide)
# ------------------------------------------------------------------------------

def find_residues_in_pocket(atom_list, centroid, distance_threshold=4.0):
    """
    Encuentra residuos que estén a < distance_threshold Å del 'centroid'.
    """
    residues_in_pocket = set()

    for atom in atom_list:
        x, y, z = atom['x'], atom['y'], atom['z']
        atom_coord = np.array([x, y, z])
        dist = np.linalg.norm(atom_coord - centroid)

        if dist <= distance_threshold:
            residues_in_pocket.add((atom['chain'], atom['res_name'], atom['res_num']))

    # Ordenamos por chain y res_num
    return sorted(residues_in_pocket, key=lambda x: (x[0], x[2]))


# ------------------------------------------------------------------------------
# 6c) Guardar en un PDB SÓLO los átomos de ciertos residuos
# ------------------------------------------------------------------------------

def save_residues_as_pdb(atom_list, residues, output_filename):
    """
    Guarda en un PDB únicamente los átomos pertenecientes a la lista 'residues',
    donde cada elemento es (chain, res_name, res_num).
    """
    residue_set = set(residues)  # para búsqueda rápida

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
# 7) Scripts de visualización (PyMOL/Chimera)
# ------------------------------------------------------------------------------

def generate_pymol_script(pdb_filename, output_script):
    """
    Crea un script de PyMOL para visualizar los pockets como esferas (resn POC).
    """
    with open(output_script, 'w') as file:
        file.write(f"load {pdb_filename}\n")
        file.write("show spheres, resn POC\n")
        file.write("color yellow, resn POC\n")
        file.write("set sphere_scale, 1.0\n")
        file.write("zoom\n")

def generate_chimera_script(original_pdb, pockets_pdb, output_script):
    """
    Crea un script para Chimera que abre la proteína y los pockets,
    y luego los muestra como esferas amarillas.
    """
    with open(output_script, 'w') as file:
        file.write(f"open {original_pdb}\n")
        file.write(f"open {pockets_pdb}\n")
        file.write("select :POC\n")
        file.write("rep sphere sel\n")
        file.write("color yellow sel\n")
        file.write("~select\n")
        file.write("focus\n")

# ------------------------------------------------------------------------------
# 8) MAIN EXECUTION FLOW
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Leer info completa de átomos
    protein_atoms = read_pdb_atoms("2lzm.pdb")
    # 2. Extraer únicamente las coords como un array numpy
    protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])

    # 3. Crear la malla y aplicar Difference of Gaussians
    grid_points = create_grid(protein_coords, spacing=0.5)
    dog_filtered = apply_dog_filter(grid_points, protein_coords)

    # 4. Extraer puntos que superan el umbral
    pocket_candidates = extract_pocket_points(grid_points, dog_filtered, threshold_percentile=95)

    # 5. Clusterizar
    pocket_clusters = cluster_pockets(pocket_candidates)

    # 6. Generar nombres de archivos únicos
    pdb_filename = get_unique_filename("predicted_pockets", "pdb")
    pymol_script = get_unique_filename("visualize_pockets", "pml")

    if len(pocket_clusters) > 0:
        # 6a. Guardar un PDB con la info (centróides)
        save_pockets_as_pdb(pocket_clusters, pdb_filename)
        print(f"✅ Pockets file saved as {pdb_filename}")

        # 6b. Para cada pocket, buscar residuos cercanos al CENTRÓIDE
        for i, pocket in enumerate(pocket_clusters, start=1):
            # Distancia a < X Å (e.g. 5.0) del centróide
            residues = find_residues_in_pocket(
                protein_atoms,
                pocket['centroid'],
                distance_threshold=5.0  # Ajusta según te convenga
            )

            # 6c. Guardar un PDB con esos residuos
            pocket_residues_pdb = get_unique_filename(f"pocket_{i}_residues", "pdb")
            save_residues_as_pdb(protein_atoms, residues, pocket_residues_pdb)
            print(f"   Pocket {i} residues saved in: {pocket_residues_pdb}")

        # 7. Generar scripts de visualización
        pymol_script = get_unique_filename("visualize_pockets", "pml")
        generate_pymol_script(pdb_filename, pymol_script)
        print(f"✅ Visualization script saved as {pymol_script} (Use in PyMOL)")

        chimera_script = get_unique_filename("visualize_pockets", "cmd")
        generate_chimera_script("2lzm.pdb", pdb_filename, chimera_script)
        print(f"✅ Visualization script for Chimera saved as {chimera_script}")

    else:
        print("❌ No significant pockets found.")
