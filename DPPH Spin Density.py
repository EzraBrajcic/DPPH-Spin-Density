#%%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Atom import atom
from Molecule import molecule
from DPPH_Lib import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import time

scale_f = 0.529177249 # Convert from a.u to Angstrom for slater exponentials

# DPPH molecule atom positions
# All positions in Angstroms using Cartesian coordinates [x, y, z]
# N1 and N2 atoms
n1_atom = atom([0.0, 0.0, 0.0], 
generate_mtps([
         (2, 0, 0, 0.301),
         (2, 1, 0, 0.001),
         (2, 1, 1, -0.014),
         (2, 1, -1, -0.006),
         (3, 2, 0, 0.134),
         (3, 2, 1, 0.027),
         (3, 2, -1, -0.016),
         (3, 2, 2, -0.018),
         (3, 2, -2, -0.016)], None),
     4.62/scale_f)

n2_atom = atom([1.337, 0.0, 0.0],
     generate_mtps([
         (2, 0, 0, 0.311),
         (2, 1, 0, 0.036),
         (2, 1, 1, 0.045),
         (2, 1, -1, 0.37),
         (3, 2, 0, 0.138),
         (3, 2, 1, -0.023),
         (3, 2, -1, -0.039),
         (3, 2, 2, 0.026),
         (3, 2, -2, 0.004)], None),
     4.18/scale_f)
    
# Euler angles in: [gamma, alpha, beta,
# Apply rotation to both nitrogen atoms
n1_rotated = n1_atom.rotate(np.pi/2, 0.105, -0.087266)
n2_rotated = n2_atom.rotate(np.pi/2, -0.10297, 0.14486)

# DPPH molecule list with rotated nitrogen atoms
DPPH = [
    n1_rotated,
    n2_rotated,
]

phenyl1_origin = [-1.68855, 2.171, -0.1]
rot_angles1 = [-np.pi/3 + 0.175, -0.1047, -0.1727]
# Create a rotatable phenyl group
phenyl1_atoms = create_phenyl(
    origin=phenyl1_origin,
    zeta=3.7/scale_f,
    # Unique multipole populations for each carbon
    multipoles=[
        generate_mtps([(2, 0, 0, -0.049)], rot_angles1),
        generate_mtps([(2, 0, 0, 0.071)], rot_angles1),
        generate_mtps([(2, 0, 0, -0.017)], rot_angles1),
        generate_mtps([(2, 0, 0, 0.078)], rot_angles1),
        generate_mtps([(2, 0, 0, -0.033)], rot_angles1),
        generate_mtps([(2, 0, 0, 0.079)], rot_angles1)])

# Rotate the phenyl group
rotated_phenyl1 = rotate_group(
    phenyl1_atoms, 
    rot_angles1,
    # Specify custom pivot point
    pivot_point=phenyl1_origin
)

# Add phenyl group to DPPH list
DPPH.extend(rotated_phenyl1)

phenyl2_origin = [-1.68855, -2.171, -0.1]
rot_angles2 = [np.pi/3 + 0.08, -0.2024, -0.087266]

# Create a rotatable phenyl group
phenyl2_atoms = create_phenyl(
    origin=phenyl2_origin,
    zeta=3.7/scale_f,
    # Unique multipole populations for each carbon
    multipoles=[
        generate_mtps([(2, 0, 0, -0.010)], rot_angles2),
        generate_mtps([(2, 0, 0, 0.031)], rot_angles2),
        generate_mtps([(2, 0, 0, -0.013)], rot_angles2),
        generate_mtps([(2, 0, 0, 0.045)], rot_angles2),
        generate_mtps([(2, 0, 0, -0.028)], rot_angles2),
        generate_mtps([(2, 0, 0, 0.038)], rot_angles2)])

# Rotate the phenyl group by different angles
rotated_phenyl2 = rotate_group(
    phenyl2_atoms, 
    rot_angles2,
    # Specify custom pivot point
    pivot_point=phenyl2_origin
)
# Add phenyl group to DPPH list
DPPH.extend(rotated_phenyl2)


# Create a picryl group with rotation
# Define picryl origin at N2 position
# Origin at C13 (central carbon of picryl)

# Define the picryl group atoms (relative positions to picryl_origin)
rot_angles3 = [-np.pi/20, 0.17104, 0.0]
picryl_atoms = [
    
    # C13 (central carbon of picryl) - at origin
    atom([0.0, 0.0, 0.0], 
        generate_mtps([(2, 0, 0, -0.012)], rot_angles3),
        3.7/scale_f),
    
    # C14 - Ortho position with NO2 group 
    atom([0.7, 1.2, 0.0], 
        generate_mtps([(2, 0, 0, 0.055)], rot_angles3),
        3.7/scale_f),
    
    # C15 - Meta position
    atom([2.1, 1.2, 0.0], 
        generate_mtps([(2, 0, 0, -0.003)], rot_angles3),
        3.7/scale_f),
    
    # C16 - Para position with NO2 group
    atom([2.8, 0.0, 0.0], 
        generate_mtps([(2, 0, 0, 0.09)], rot_angles3),
        3.7/scale_f),
    
    # C17 - Meta position
    atom([2.1, -1.2, 0.0], 
        generate_mtps([(2, 0, 0, -0.026)], rot_angles3),
        3.7/scale_f),
    
    #C18 - Ortho position with NO2 group    
    atom([0.7, -1.2, 0.0], 
        generate_mtps([(2, 0, 0, 0.032)], rot_angles3),
        3.7/scale_f),   
     
    # NO2 groups on picryl
    # N3 (ortho nitro group)
    atom([0.0, 2.42, 0.0], 
        generate_mtps([(2, 0, 0, -0.011)], rot_angles3),
        4.231),
    
    # N4 (para nitro group)
    atom([4.2, 0.0, 0.0], 
        generate_mtps([(2, 0, 0, -0.008)], rot_angles3),
        4.231/scale_f),
    
    # N5 (ortho nitro group)
    atom([0.0, -2.42, 0.0], 
        generate_mtps([(2, 0, 0, -0.003)], rot_angles3),
        4.231/scale_f),
        
    # Add oxygen atoms for the nitro groups
    # O1, O2 - oxygens for N3 (ortho nitro group)
    atom([0.7, 3.15, 0.5], 
        generate_mtps([(2, 0, 0, 0.012)], rot_angles3),
        4.762/scale_f),
    atom([-0.7, 3.15, -0.5], 
        generate_mtps([(2, 0, 0, 0.004)], rot_angles3),
        4.762/scale_f),
        
    # O3, O4 - oxygens for N4 (para nitro group)
    atom([4.8, 0.95, 0.0], 
        generate_mtps([(2, 0, 0, 0.009)], rot_angles3),
        4.762/scale_f),
    atom([4.8, -0.95, 0.0], 
        generate_mtps([(2, 0, 0, 0.006)], rot_angles3),
        4.762/scale_f),
        
    # O5, O6 - oxygens for N5 (ortho nitro group)
    atom([0.7, -3.15, 0.5], 
        generate_mtps([(2, 0, 0, -0.001)], rot_angles3),
        4.762/scale_f),
    atom([-0.7, -3.15, -0.5], 
        generate_mtps([(2, 0, 0, 0.026)], rot_angles3),
        4.762/scale_f),
    
    # H11, H12 - Picryl hydrogens for C16 and C17
    atom([2.8666, 1.9666, 0.0],
         generate_mtps([(2, 0, 0, -0.026)], rot_angles3),
         0.72/scale_f),
    atom([2.8666, -1.9666, 0.0],
        generate_mtps([(2, 0, 0, 0.005)], rot_angles3),
        0.72/scale_f)
    ]

# Create the initial positioned picryl group
picryl_origin = [1.337+1.355, 0.0, 0.0]
positioned_picryl = []
for picryl_atom in picryl_atoms:
    x, y, z = picryl_atom.origin_cartesian
    # Adjust position relative to the picryl origin
    positioned_atom = atom(
        [x + picryl_origin[0], y + picryl_origin[1], z + picryl_origin[2]],
        picryl_atom.mpts,
        picryl_atom.zeta
    )
    positioned_picryl.append(positioned_atom)

# Rotate the picryl group around the z-axis
rotated_picryl = rotate_group(
    positioned_picryl,
    rot_angles3,
    pivot_point=[1.337, 0.0, 0.0]  # N2 position as pivot
)

# Add the rotated picryl group to DPPH list
DPPH.extend(rotated_picryl)

# Add hydrogens to phenyl groups
phenyl1_Hs = add_hydrogen(rotated_phenyl1, zeta = 0.72/scale_f, 
                          multipoles = [generate_mtps([(2, 0, 0, 0.079)], rot_angles1),
                                        generate_mtps([(2, 0, 0, -0.033)], rot_angles1),
                                        generate_mtps([(2, 0, 0, 0.078)], rot_angles1),
                                        generate_mtps([(2, 0, 0, -0.017)], rot_angles1),
                                        generate_mtps([(2, 0, 0, 0.071)], rot_angles1)
                                        ])
# Add hydrogens to DPPH list
DPPH.extend(phenyl1_Hs)

phenyl2_Hs = add_hydrogen(rotated_phenyl2, zeta = 0.72/scale_f, 
                          multipoles = [generate_mtps([(2, 0, 0, 0.038)], rot_angles1),
                                        generate_mtps([(2, 0, 0, -0.028)], rot_angles1),
                                        generate_mtps([(2, 0, 0, 0.045)], rot_angles1),
                                        generate_mtps([(2, 0, 0, -0.011)], rot_angles1),
                                        generate_mtps([(2, 0, 0, 0.031)], rot_angles1)
                                        ])
# Add hydrogens to DPPH list
DPPH.extend(phenyl2_Hs)

# Define DPPH molecule object from previously defined DPPH list
DPPH_molecule = molecule(DPPH)

# Define atom types for plotting
atom_types = [
    'N', 'N',                                  # N1, N2
    'C', 'C', 'C', 'C', 'C', 'C',              # First phenyl ring (C1-C6)
    'C', 'C', 'C', 'C', 'C', 'C',              # Second phenyl ring (C7-C12)
    'C', 'C', 'C', 'C', 'C', 'C',              # Picryl ring (C13-C18)
    'N', 'N', 'N',                             # NO2 groups (N3-N5)
    'O', 'O', 'O', 'O', 'O', 'O'               # Oxygen atoms (O1-O6)
]

# Define atom labels for plotting
atom_labels = [
    'N1', 'N2',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
    'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
    'N3', 'N4', 'N5',
    'O1', 'O2', 'O3', 'O4', 'O5', 'O6'
]

# Define bond connections (pairs of atom indices that should be connected)
bonds = [
    # N-N hydrazyl bond
    (0, 1),
    
    # First phenyl ring connections
    (0, 2),  # N1-C1
    (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2),  # C1-C2-C3-C4-C5-C6-C1
    
    # Second phenyl ring connections
    (0, 8),  # N1-C7
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8),  # C7-C8-C9-C10-C11-C12-C7
    
    # Picryl group connections
    (1, 14),  # N2-C13
    (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 14),  # C13-C14-C15-C16-C17-C18-C13
    
    # NO2 connections
    (15, 20),  # C14-N3
    (17, 21),  # C16-N4
    (19, 22),   # C18-N5
    
    # Oxygen connections to nitrogens
    (20, 23), (20, 24),  # N3-O1, N3-O2
    (21, 25), (21, 26),  # N4-O3, N4-O4
    (22, 27), (22, 28)   # N5-O5, N5-O6
]


# Visualization parameters
bounds = (-9, 9) # Force x and y range to be identical to retain rotational invariance
x_range = bounds
y_range = bounds
z_slice = 0.0
resolution = 5000  # Resolution for visualization, 5000 means a 5000 by 5000 grid within the range of of bounds
name1 = 'DPPH_spin_density.png'
name2 = 'DPPH_spin_density_contour.png'
dpi = 800 # Plot resoultion

# Create visualization with both 2D slice and 3D view
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig1)

# Create 2D grid in XY plane for the slice
x_vals = cp.linspace(x_range[0], x_range[1], resolution)
y_vals = cp.linspace(y_range[0], y_range[1], resolution)
x_grid, y_grid = cp.meshgrid(x_vals, y_vals)

# Create 3D grid with constant z value (for a slice)
z_grid = cp.ones_like(x_grid) * z_slice

# Convert Cartesian to spherical coordinates
r_cp = cp.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
theta_cp = cp.arccos(z_grid / (r_cp + 1e-10))  # Add small epsilon to avoid division by zero
phi_cp = cp.arctan2(y_grid, x_grid)

# Create position arrays
pos = [r_cp, theta_cp, phi_cp]

# Calculate spin densities
clockStart = time.time()
spin_density = DPPH_molecule.spin_density_eval_batched(pos)
spin_density_np = cp.asnumpy(spin_density)
print(f'Time elapsed: {time.time() - clockStart}\n')

# Dictionary of atom colors and sizes
atom_colors = {
    'N': 'blue',
    'C': 'grey',
    'O': 'red'
}
atom_sizes = {
    'N': 150,
    'C': 150,
    'O': 150
}

# Plot 2D spin density slice
ax1 = fig1.add_subplot(gs[0, 0])
im = ax1.imshow(spin_density_np, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
               origin='lower', cmap='magma', interpolation='spline16')
ax1.set_xlabel('X Position (Å)')
ax1.set_ylabel('Y Position (Å)')

# Center the molecule in the plot by finding the center of mass
x_coords = [atom.origin_cartesian[0] for atom in DPPH_molecule.atoms]
y_coords = [atom.origin_cartesian[1] for atom in DPPH_molecule.atoms]
center_x = sum(x_coords) / len(x_coords)
center_y = sum(y_coords) / len(y_coords)

# Calculate the distance from center to farthest atom to determine required plotting range
max_distance = max([np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in zip(x_coords, y_coords)])
padding = 1.2
plot_range = max_distance * padding

# Set equal axis limits centered around the molecular center
ax1.set_xlim(center_x - plot_range, center_x + plot_range)
ax1.set_ylim(center_y - plot_range, center_y + plot_range)

# Ensure equal scaling (1:1 aspect ratio)
ax1.set_aspect('equal')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Find the minimum and maximum values of the spin density for colourbar range
spin_density_min = np.min(spin_density_np)
spin_density_max = np.max(spin_density_np)

# Create ticks for the colourbar (e.g., 10 ticks)
ticks = np.linspace(spin_density_min, spin_density_max, 10)

# Colourbar
cbar = plt.colorbar(im, cax=cax, label=r'Spin Density (Normalized to 1$\cdot\mu_B$)', ticks=ticks)
cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

# Draw bonds
for bond in bonds:
    
    atom1 = DPPH_molecule.atoms[bond[0]]
    atom2 = DPPH_molecule.atoms[bond[1]]
    
    x1, y1, z1 = atom1.origin_cartesian
    x2, y2, z2 = atom2.origin_cartesian
    
    # Draw bonds in 2D slice
    ax1.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5)
    
# Draw atoms
for idx, (atom_obj, label) in enumerate(zip(DPPH_molecule.atoms, atom_labels)):
    x, y, z = atom_obj.origin_cartesian
    atom_type = label[0]
    color = atom_colors.get(atom_type, 'white')
    size = atom_sizes.get(atom_type, 100)
    
    ax1.plot(x, y, 'o', markersize=3, color=color, markeredgecolor='black')
    ax1.text(x+0.1, y+0.1, label, color='white', fontsize=6)
        
plt.tight_layout()
plt.savefig(name1, dpi=dpi)
plt.show()  

# Plot 2D spin density slice
fig2 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig2)
axc = fig2.add_subplot(gs[0, 0])

im = axc.imshow(spin_density_np, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
               origin='lower', cmap='magma', interpolation='spline16')

# Add contour lines on top of the colourmap
contour = axc.contour(x_vals.get(), y_vals.get(), spin_density_np, 
                      levels=10, 
                      colors='white',
                      linewidths=0.3,
                      alpha=0.7)

axc.set_xlabel('X Position (Å)')
axc.set_ylabel('Y Position (Å)')

# Center the molecule in the plot by finding the center of mass
x_coords = [atom.origin_cartesian[0] for atom in DPPH_molecule.atoms]
y_coords = [atom.origin_cartesian[1] for atom in DPPH_molecule.atoms]
center_x = sum(x_coords) / len(x_coords)
center_y = sum(y_coords) / len(y_coords)

# Calculate the distance from center to farthest atom to determine required plotting range
max_distance = max([np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in zip(x_coords, y_coords)])
padding = 1.2
plot_range = max_distance * padding

# Set equal axis limits centered around the molecular center
axc.set_xlim(center_x - plot_range, center_x + plot_range)
axc.set_ylim(center_y - plot_range, center_y + plot_range)

# Ensure equal scaling (1:1 aspect ratio)
axc.set_aspect('equal')

# Find the minimum and maximum values of the spin density for colorbar range
spin_density_min = np.min(spin_density_np)
spin_density_max = np.max(spin_density_np)

# Create ticks for the colourbar (e.g., 10 ticks)
ticks = np.linspace(spin_density_min, spin_density_max, 10)

# Colourbar
divider = make_axes_locatable(axc)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, label=r'Spin Density (Normalized to 1$\cdot\mu_B$)', ticks=ticks)
cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

# Draw bonds
for bond in bonds:
    
    atom1 = DPPH_molecule.atoms[bond[0]]
    atom2 = DPPH_molecule.atoms[bond[1]]
    
    x1, y1, z1 = atom1.origin_cartesian
    x2, y2, z2 = atom2.origin_cartesian
    
    # Draw bonds in 2D slice
    axc.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5)
    
# Draw atoms
for idx, (atom_obj, label) in enumerate(zip(DPPH_molecule.atoms, atom_labels)):
    x, y, z = atom_obj.origin_cartesian
    atom_type = label[0]
    color = atom_colors.get(atom_type, 'white')
    size = atom_sizes.get(atom_type, 100)
    
    axc.plot(x, y, 'o', markersize=3, color=color, markeredgecolor='black')
    axc.text(x+0.1, y+0.1, label, color='white', fontsize=6)
        
plt.tight_layout()
plt.savefig(name2, dpi=dpi)
plt.show()  

fig3 = plt.figure(figsize=(8, 8))              
gs = GridSpec(1, 1, figure=fig3)

# 3D molecule visualization
ax2 = fig3.add_subplot(gs[0, 0], projection='3d')
ax2.set_title('DPPH Molecular Structure')
ax2.set_xlabel('X Position (Angstrom)')
ax2.set_ylabel('Y Position (Angstrom)')
ax2.set_zlabel('Z Position (Angstrom)')

# Find the center in 3D
z_coords = [atom.origin_cartesian[2] for atom in DPPH_molecule.atoms]
center_z = sum(z_coords) / len(z_coords)

# Find the maximum distance in 3D
max_distance_3d = max([np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2) 
                       for x, y, z in zip(x_coords, y_coords, z_coords)])
plot_range_3d = max_distance_3d * padding

# Set limits centered around the molecular center
ax2.set_xlim(center_x - plot_range_3d, center_x + plot_range_3d)
ax2.set_ylim(center_y - plot_range_3d, center_y + plot_range_3d)
ax2.set_zlim(center_z - plot_range_3d, center_z + plot_range_3d)

# Make the aspect ratio equal for rotational invariance
ax2.set_box_aspect([1, 1, 1])

# Draw bonds
for bond in bonds:
    atom1 = DPPH_molecule.atoms[bond[0]]
    atom2 = DPPH_molecule.atoms[bond[1]]
    
    x1, y1, z1 = atom1.origin_cartesian
    x2, y2, z2 = atom2.origin_cartesian
    
    # Draw bonds in 3D
    ax2.plot([x1, x2], [y1, y2], [z1, z2], 'k-', linewidth=1.5, alpha=0.7)

# Draw atoms in 3D
for idx, (atom_obj, label) in enumerate(zip(DPPH_molecule.atoms, atom_labels)):
    x, y, z = atom_obj.origin_cartesian
    atom_type = label[0]
    color = atom_colors.get(atom_type, 'white')
    size = atom_sizes.get(atom_type, 100)
    
    # 3D plot - show all atoms
    ax2.scatter(x, y, z, color=color, s=size, edgecolor='black', alpha=0.8)
    
    # Only label important atoms in 3D view to avoid clutter
    if atom_type == 'N' or label in ['C1', 'C7', 'C13']:
        ax2.text(x+0.1, y+0.1, z+0.1, label, fontsize=8)

# Change view angle
ax2.view_init(elev = 35, azim = -45)
plt.tight_layout()
plt.show()
#%%