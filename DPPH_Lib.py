#%%
import numpy as np
import numpy as np
from Atom import atom
from scipy.spatial.transform import Rotation as R

def generate_mtps(mpts, expand = None):
    """
    Generates multipole population dictionary for atom depending on quantum numbers
    n, l, and m. If only a single multipole moment is given, higher order
    populations are generated based on the appendix from J.X. Boucherle et al. (1987)
    when the expand argument is provided.

    Args:
        mpts (list(dict)): Dictionary of multipole populations for atom based on quantum numbers
        expand (list, optional): Set of euler angles that define higher order multipole populations. Defaults to None.

    Returns:
        dict: List of multipole dictionaries for atom
    """
    
    # Compute multipole expansion if expand argument is provided
    if expand is not None:
        expanded_mpts = []
        
        for n, l, m, coeff in mpts:
            expanded_mpts.append((n, l, m, coeff))
        
        expanded_mpts.append((3, 2, 0, mpts[0][3] * (((expand[0]**2)/2 - 1/6) * 2/np.sqrt(3))))
        expanded_mpts.append((3, 2, 1, mpts[0][3] * 2 * expand[1] * expand[0] / np.pi))
        expanded_mpts.append((3, 2, -1, mpts[0][3] * 2 * expand[2] * expand[0] / np.pi))
        expanded_mpts.append((3, 2, 2, mpts[0][3] * 2 * expand[1] * expand[2] / np.pi))
        expanded_mpts.append((3, 2, -2, mpts[0][3] * (expand[1]**2 - expand[2]**2) / np.pi))
        mpts = expanded_mpts
        
    return {(n, l, m): coeff for n, l, m, coeff in mpts}

def create_phenyl(origin, zeta=3.7, multipoles=None):
    """
    Create a phenyl group centered at a specific origin with rotation capabilities.
    
    Parameters:
    origin (list): [x, y, z] coordinates for the center of the phenyl ring
    zeta (float): Orbital exponent for the STO, set to 3.7 for carbons
    multipoles (list): List of multipole dictionaries for each carbon atom
    rotation_angles (list): [gamma, alpha, beta] Euler angles in radians for rotation
    
    Returns:
    list: List of atom objects representing the phenyl ring
    """
    # Standard benzene ring has 6 carbon atoms at 60° intervals
    # with typical C-C bond length of ~1.4Å
    ring_radius = 1.4
    
    # Generate base carbon positions (planar hexagon in XY plane)
    carbon_positions = []
    for i in range(6):
        angle = i * (2 * np.pi / 6)  # 60° intervals in radians
        x = ring_radius * np.cos(angle)
        y = ring_radius * np.sin(angle)
        z = 0.0  # Planar ring in XY plane
        carbon_positions.append([x, y, z])
    
    # Create atom objects for the phenyl group
    phenyl_atoms = []
    for idx, pos in enumerate(carbon_positions):
        
        # Adjust position relative to desired origin in the molecule
        adjusted_pos = [
            pos[0] + origin[0],
            pos[1] + origin[1], 
            pos[2] + origin[2]
        ]
        
        # Create atom with position and parameters
        carbon_atom = atom(
            adjusted_pos,
            multipoles[idx],
            zeta
        )
        phenyl_atoms.append(carbon_atom)
    
    return phenyl_atoms

def rotate_group(group_atoms, rotation_angles, pivot_point=None):
    """
    Rotate a group of atoms around a pivot point.
    
    Parameters:
    group_atoms (list): List of atom objects to rotate
    rotation_angles (list): [gamma, alpha, beta] Euler angles in radians
    pivot_point (list, optional): [x, y, z] point to rotate around, defaults to center of atoms
    
    Returns:
    list: Updated atom objects after rotation
    """
    # If no pivot point provided, calculate center of the group
    if pivot_point is None:
        x_sum = y_sum = z_sum = 0
        for atom_obj in group_atoms:
            x, y, z = atom_obj.origin_cartesian
            x_sum += x
            y_sum += y
            z_sum += z
        pivot_point = [x_sum/len(group_atoms), y_sum/len(group_atoms), z_sum/len(group_atoms)]
    
    # Create rotation object
    gamma, alpha, beta = rotation_angles
    rotation = R.from_euler('zyz', [gamma, alpha, beta])
    
    # Apply rotation to each atom
    rotated_group = []
    for atom_obj in group_atoms:
        # Get current position
        x, y, z = atom_obj.origin_cartesian
        
        # Shift to origin (relative to pivot)
        x_rel = x - pivot_point[0]
        y_rel = y - pivot_point[1]
        z_rel = z - pivot_point[2]
        
        # Apply rotation
        pos_rel = np.array([[x_rel, y_rel, z_rel]])
        rotated_pos = rotation.apply(pos_rel)[0]
        
        # Shift back to original position
        new_pos = [
            rotated_pos[0] + pivot_point[0],
            rotated_pos[1] + pivot_point[1],
            rotated_pos[2] + pivot_point[2]
        ]
        
        # Create new atom with rotated position and same properties
        new_atom = atom(
            new_pos,
            atom_obj.mpts,
            atom_obj.zeta
        )
        
        # Rotate the multipole terms using the same rotation angles
        new_atom = new_atom.rotate(*rotation_angles)
        
        rotated_group.append(new_atom)
    
    return rotated_group

def add_hydrogen(phenyl, zeta = 1.0, multipoles = None):
    """
    Adds hydrogen atoms to a phenyl group

    Args:
        phenyl (list): list of carbon atoms in phenyl
        zeta (float, optional): Orbital exponent for the STO. Defaults to 1.0.
        multipoles (list, optional): List of multipole dictionaries for each hydrogen. Defaults to None.

    Returns:
       list: list of hydrogen atom objects oriented radially outward from the phenyl group's plane at a distance of 1.09 Angstrom
    """
    
    hydrogen_positions = []
    
    # Average phenyl hydrogen bond length
    hydrogen_distance = 1.09
    
    # Creates list of hydrogen atom positions relative to each carbon atom in the phenyl group
    # except for the carbon bonding the phenyl group to the next group, 
    # for DPPH it's bonded to the hydrazyl group
    for n in range(1, len(phenyl)):
        r = phenyl[n].origin[0] + hydrogen_distance
        theta = phenyl[n].origin[1]
        phi = phenyl[n].origin[2]
        hydrogen_positions.append([r, theta, phi])
        
    # Create atom objects for the hydrogens
    hydrogen_atoms = []
    for idx, pos in enumerate(hydrogen_positions):
        
        # Adjust position relative to desired origin in the molecule
        adjusted_pos = [
            pos[0] * np.sin(pos[1]) * np.cos(pos[2]),
            pos[0] * np.sin(pos[1]) * np.sin(pos[2]),
            pos[0] * np.cos(pos[1])
        ]
        
        # Create hydrogen atom with position and parameters
        hydrogen_atom = atom(
            adjusted_pos,
            multipoles[idx],
            zeta)
        hydrogen_atoms.append(hydrogen_atom)
    return hydrogen_atoms