import cupy as cp

class molecule:
    
    def __init__(self, atoms):
        """
        Initialize a molecule with a list of atoms.
        
        Parameters:
        atoms (list): List of atom objects
        """
        self.atoms = atoms
    
    def spin_density_eval(self, pos):
        """
        Compute spin density using direct summation of multipolar contributions
        from each atom, parallelized with CuPy.
        
        Parameters:
        pos (list): [r, theta, phi] arrays in spherical coordinates
        
        Returns:
        cupy.ndarray: Spin density values
        """
        print("Calculating spin density...")
        
        # Initialize spin density grid
        r_grid, theta_grid, phi_grid = pos
        
        # Pre-allocate an array to store all atom contributions in one batch
        # Shape: [num_atoms, grid_shape...]
        grid_shape = r_grid.shape
        atom_densities = cp.zeros((len(self.atoms), *grid_shape), dtype = cp.float32)
        
        # Calculate each atom's contribution in parallel
        for i, atom_obj in enumerate(self.atoms):
            print(f"Processing atom {i+1}/{len(self.atoms)}...")
            atom_densities[i] = atom_obj.spin_density(pos)
        
        # Sum along the atom dimension to get total spin density
        # This is performed as a single GPU operation
        spin_density = cp.sum(atom_densities, axis = 0)
        
        print("Summation of spin density completed.")
        return spin_density
    
    def spin_density_eval_batched(self, pos, batch_size = 5):
        """
        Compute spin density using batch processing to reduce vram usage.
        This is useful for very large grids or many atoms since when vram usage exceeds the
        vram of the GPU, it gets sent to the CPU's ram which is much slower and has far higher
        latency due to how far away it is physically from the GPU.
        
        Reduces vram usage by ~50% compared to non-batched process.
        
        Parameters:
        pos (list): [r, theta, phi] arrays in spherical coordinates
        batch_size (int): Number of atoms to process in each batch
        
        Returns:
        numpy.ndarray or cupy.ndarray: Spin density values
        """
        print("Calculating spin density with batch processing...")
        r_grid, theta_grid, phi_grid = pos
        
        # Initialize spin density grid
        spin_density = cp.zeros_like(r_grid)
        
        # Process atoms in batches
        num_atoms = len(self.atoms)
        for batch_start in range(0, num_atoms, batch_size):
            batch_end = min(batch_start + batch_size, num_atoms)
            print(f"Processing atom batch {batch_start+1}-{batch_end}/{num_atoms}...")
            
            # Process batch of atoms
            batch_atoms = self.atoms[batch_start:batch_end]
            batch_size_actual = len(batch_atoms)
            
            # Pre-allocate array for current batch
            batch_densities = cp.zeros((batch_size_actual, *r_grid.shape), dtype=cp.float32)
            
            # Calculate each atom's contribution in current batch
            for i, atom_obj in enumerate(batch_atoms):
                batch_densities[i] = atom_obj.spin_density(pos)
            
            # Add batch sum to total
            spin_density += cp.sum(batch_densities, axis=0)
            
            # Force synchronization and memory cleanup
            cp.cuda.stream.get_current_stream().synchronize()
            del batch_densities
            cp.get_default_memory_pool().free_all_blocks()
        
        print("Spin density calculation completed.")
        return spin_density