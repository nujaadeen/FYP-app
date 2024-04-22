import networkx as nx
import numpy as np
from biopandas.pdb import PandasPdb
import periodictable
from esmEncoder import ESMEncoder

# Torch
import torch

# Torch Geometric
import torch_geometric


def pdb_to_graph(pdb_location, pdb_type='complex', heavy_chain='A', light_chain='B', virus='C'):
    """Returns torch_geometric graph data object from a .pdb file
    
    Graph with resiudes as nodes is returned when node Arg is 'Residue'. Residue features are ESM
    encodings of residues.  Distance between nodes are the edge attributes in both cases.

    Args:
        pdb_file (str): Location of the .pdb file
        pdb_type (str): 'complex' or 'separate'.

    Returns:
        networkx.classes.graph.Graph: Atomic/Residual Graph for .pdb file
    """
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    
    esm_encoder = ESMEncoder()

    # Read pdb files
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_location)
    
    # Use NetworkX to create a graph from the atom coordinates
    graph = nx.Graph()

    residues_and_coords = ppdb.df['ATOM'][['element_symbol', 'residue_number','x_coord', 'y_coord', 'z_coord']].values
    all_residues = ppdb.df['ATOM'][['residue_number']].values
    residue_names = ppdb.df['ATOM'][['residue_name']].values
    chain_ids = ppdb.df['ATOM'][['chain_id']].values
    residues = np.unique(all_residues)

    if pdb_type == 'complex':
        ab_seq = d[residue_names[0][0]]
        virus_seq = ''
        for i in range(len(all_residues)-1):
            if chain_ids[i+1] == heavy_chain or chain_ids[i+1] == light_chain:
                if chain_ids[i] != chain_ids[i+1]:
                    ab_seq += ' '
                if all_residues[i] != all_residues[i+1]:
                    ab_seq += d[residue_names[i+1][0]]
            else:
                if chain_ids[i] != chain_ids[i+1]:
                    virus_seq += d[residue_names[i][0]]
                if all_residues[i] != all_residues[i+1]:
                    virus_seq += d[residue_names[i+1][0]]

        ab_embedding = esm_encoder(ab_seq)
        virus_embedding = esm_encoder(virus_seq)

        # This list is made to find indices except for space values in the sequence used to differenciate chains
        ab_indices = [] 
        count = 0
        for idx, residue in enumerate(ab_seq):
            if residue == ' ':
                count += 1
            ab_indices.append(idx+count)

        # This list is made to find indices except for space values in the sequence used to differenciate chains
        virus_indices = [] 
        count = 0
        for idx, residue in enumerate(virus_seq):
            if residue == ' ':
                count += 1
            virus_indices.append(idx+count)
        
        indices = ab_indices + virus_indices
    
    elif pdb_type == 'separate':
        seq = d[residue_names[0][0]]
        for i in range(len(all_residues)-1):
            if chain_ids[i] != chain_ids[i+1]:
                seq += ' '
            if all_residues[i] != all_residues[i+1]:
                seq += d[residue_names[i+1][0]]

        # This list is made to find indices except for space values in the sequence used to differenciate chains
        indices = [] 
        count = 0
        for idx, residue in enumerate(seq):
            if residue == ' ':
                count += 1
            indices.append(idx+count)

        embedding = esm_encoder(seq)

    else:
        raise Exception("pdb_type should be \'complex\' or \'separate\'")

    
    residue_level_list = [] # Same as residues_and _coords with atoms of same residue grouped as a list (2D)
    same_residue_list = []

    for residue_number_1 in residues:
        for residue_and_coord in residues_and_coords:
            residue_number_2 = int(residue_and_coord[1])
            if residue_number_1 == residue_number_2:
                same_residue_list.append(residue_and_coord)
        residue_level_list.append(same_residue_list)
        same_residue_list = []

    residue_details = [] # [Number, Name, Chaind ID] of all atoms
    for i in range (len(residue_names)):
        residue_details.append([all_residues[i][0], residue_names[i][0], chain_ids[i][0]])

    residue_names_list = [] # Unique, same as residue_details 
    for residue_detail in residue_details: 
        if residue_detail not in residue_names_list:
            residue_names_list.append(residue_detail)

    # residue_names_list = np.delete(residue_names_list,0,1) # [Number, Name, Chain ID] -> [Number, Chain ID]

    residues_with_coordinates = []

    for residue in residue_level_list:
        coords = [0, 0, 0]
        mass = 0
        for atom in residue:
            atomic_name = atom[0]
            atomic_coords = atom[2:]
            atomic_mass = periodictable.elements.symbol(atomic_name).mass
            coords += atomic_mass * atomic_coords
            mass += atomic_mass
        residues_with_coordinates.append(coords / mass)

    residual_coordinates = np.array(residues_with_coordinates)
    coords = residual_coordinates

    
    # Add nodes to the graph
    num_residues = len(coords)
    for i in range(num_residues):
        node = f"{residue_names_list[i][2]}_{residue_names_list[i][1]}_{residue_names_list[i][0]}"
        node = i
        chain_id = residue_names_list[i][2]
        graph.add_node(node)
        graph.nodes[node]['x'] = coords[i][0]
        graph.nodes[node]['y'] = coords[i][1]
        graph.nodes[node]['z'] = coords[i][2]
        if pdb_type == 'complex':
            if chain_id == heavy_chain or chain_id == light_chain:
                graph.nodes[node]['Encoding'] = ab_embedding[indices[i]].tolist()
            else:
                graph.nodes[node]['Encoding'] = virus_embedding[indices[i]].tolist()
        elif pdb_type == 'separate':
            graph.nodes[node]['Encoding'] = embedding[indices[i]].tolist()
        else:
            raise Exception("pdb_type should be \'complex\' or \'separate\'")
        
        
        if (residue_names_list[i][2] == heavy_chain or residue_names_list[i][2] == light_chain):
            graph.nodes[node]['Type'] = 1
        else:
            graph.nodes[node]['Type'] = 0

    # Add edges to the graph
    for i in range(num_residues):
        for j in range(i+1, num_residues):
            dist = ((coords[i]-coords[j])**2).sum()**0.5
            if dist < 10:
                graph.add_edge(i, j, distance=dist)

    # Convert to torch geometric data object (Networkx graph => Torch Geometric Data)
    data = torch_geometric.utils.from_networkx(graph)

    # Preprocess data object 
    data.x = torch.cat((data.Encoding, torch.reshape(data.x, (data.num_nodes, 1)), torch.reshape(data.y, (data.num_nodes, 1)), 
                            torch.reshape(data.z, (data.num_nodes, 1)), torch.reshape(data.Type, (data.num_nodes, 1))), dim=1)
    
    # Convert node/edge/graph attribute tensors to same dtype: torch.float32
    data.x = data.x.to(torch.float32)
    data.edge_attr = data.distance.to(torch.float32)
    data.edge_index = data.edge_index
    
    # Remove unwanted attributes
    delattr(data, 'z') # Concatenated in Node feature x
    delattr(data, 'Encoding') # do
    delattr(data, 'Type') # do
    delattr(data, 'distance') # Renamed to edge_attr

    return data