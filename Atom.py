import numpy as np
import cupy as cp
from scipy.special import factorial
from sympy.physics.quantum.spin import WignerD
import cupyx.scipy.special._sph_harm as cupyx_special

class atom:
    
    def __init__(self, origin, mpts, zeta):
        """
        Initialize an atom with specified parameters.
        
        Parameters:
        origin (list): [x, y, z] coordinates in Cartesian
        mpts (dict): Multipole terms as {(n, l, m): coefficient} pairs
        zeta (float): Orbital exponent for the STO
        """
        self.origin_cartesian = origin  # [x, y, z]
        
        # Convert to spherical
        x, y, z = origin
        r = cp.sqrt(x**2 + y**2 + z**2)
        theta = cp.arccos(z / (r + 1e-10))
        phi = cp.arctan2(y, x)
        self.origin = [r, theta, phi]

        self.mpts = mpts
        self.zeta = zeta
    
    def sto(self, r, n):
        """
        Compute Slater-Type Orbital radial function.
        
        Parameters:
        r (numpy.ndarray or cupy.ndarray): Radial distances
        n (int): Principal quantum number for the STO
        
        Returns:
        numpy.ndarray or cupy.ndarray: STO values at given radial distances
        """
        prefactor = ((2 * self.zeta) ** (n + 0.5)) / cp.sqrt(factorial(2 * n))
        return prefactor * (r ** (n - 1)) * cp.exp(-self.zeta * r)
    
    def rotate(self, gamma, alpha, beta):
        """
        Rotate the atom's multipole terms using Wigner D-matrices.
        
        Parameters:
        alpha, beta, gamma (float): Euler angles in radians for rotation
        
        Returns:
        atom: A new atom with rotated multipole terms
        """
        # Create a new dictionary for the rotated multipole terms
        rotated_mpts = {}
        
        # For each multipole term
        for (n, l, m), coeff in self.mpts.items():
            
            # Mix all m values for a given l
            for mp in range(-l, l+1):
                
                # Calculate the Wigner D-matrix element
                d_element = complex(WignerD(l, m, mp, gamma, alpha, beta).doit())
                
                # Since we're working with real coefficients, take the real part
                d_element_real = float(d_element.real)
                
                # Update the coefficient for the (n, l, mp) term
                key = (n, l, mp)
                if key in rotated_mpts:
                    rotated_mpts[key] += coeff * d_element_real
                else:
                    rotated_mpts[key] = coeff * d_element_real
        
        # Create a new atom with the rotated multipole terms
        # Use original Cartesian coordinates instead of spherical
        return atom(self.origin_cartesian, rotated_mpts, self.zeta)
    
    def spin_density(self, pos, chunk_size = 1000000):
        """
        Compute the probability density at given positions using multipole expansion from J.X. Boucherle et al. (1987).
        Optimized with CuPy for parallel computation and chunking for memory efficiency.
        While larger chunks are usually more optimal due to fewer calls to the gpu to clean
        memory and cache. If chunck size is too large, all of the vram available to the GPU
        could be used, forcing data to be stored in the CPU's ram which is much slower and
        has far higher latency due to how far away it is physically from the GPU.
        
        Parameters:
        pos (list): [r, theta, phi] arrays in spherical coordinates
        chunk_size (int): Maximum number of points to process at once
        
        Returns:
        cupy.ndarray: Probability density values
        """
        r_grid, theta_grid, phi_grid = pos
        
        # Store original shape for later reshaping
        original_shape = r_grid.shape
        
        # Flatten input arrays for chunking
        r_flat = r_grid.flatten()
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        
        # Get total number of grid points
        total_points = r_flat.size
        
        # Initialize flattened result array
        prob_density_flat = cp.zeros(total_points, dtype=r_flat.dtype)
        
        # Process data in chunks to manage memory usage
        for start_idx in range(0, total_points, chunk_size):
            # Get end index for current chunk
            end_idx = min(start_idx + chunk_size, total_points)
            
            # Extract current chunk data
            r_chunk = r_flat[start_idx:end_idx]
            theta_chunk = theta_flat[start_idx:end_idx]
            phi_chunk = phi_flat[start_idx:end_idx]
            
            # Convert input spherical coordinates to Cartesian for this chunk
            x_chunk = r_chunk * cp.sin(theta_chunk) * cp.cos(phi_chunk)
            y_chunk = r_chunk * cp.sin(theta_chunk) * cp.sin(phi_chunk)
            z_chunk = r_chunk * cp.cos(theta_chunk)
            
            # Calculate distance from atom origin in Cartesian coordinates
            dx = x_chunk - self.origin_cartesian[0]
            dy = y_chunk - self.origin_cartesian[1]
            dz = z_chunk - self.origin_cartesian[2]
            
            # Compute distance
            distance = cp.sqrt(dx**2 + dy**2 + dz**2)
            
            # Compute new spherical coordinates relative to atom center
            r_rel = distance
            theta_rel = cp.arccos(dz / (r_rel + 1e-10))  # Small epsilon value to avoid division by zero
            phi_rel = cp.arctan2(dy, dx)
            
            # Initialize result for this chunk
            chunk_result = cp.zeros_like(r_chunk)
            
            # Dictionary to cache SphericalHarmonic calculations
            sph_harm_cache = {}
            
            # Add contributions from all multipole terms
            for (n, l, m), P_lm in self.mpts.items():
                # Cache key for spherical harmonics
                cache_key = (l, m)
                
                # Calculate or retrieve spherical harmonic values
                if cache_key not in sph_harm_cache:
                    y_lm = cupyx_special.sph_harm(m, l, phi_rel, theta_rel).real
                    sph_harm_cache[cache_key] = y_lm
                else:
                    y_lm = sph_harm_cache[cache_key]
                
                # Calculate STO part
                sto_vals = self.sto(r_rel, n)
                
                # Add contribution from this (n,l,m) term
                chunk_result += P_lm * sto_vals * y_lm
            
            # Store result for this chunk in the output array
            prob_density_flat[start_idx:end_idx] = chunk_result
            
            # Free cache memory
            del sph_harm_cache
            
            # Force memory cleanup
            cp.cuda.Stream.null.synchronize()
        
        # Reshape result to original grid shape
        prob_density = prob_density_flat.reshape(original_shape)
        
        return prob_density