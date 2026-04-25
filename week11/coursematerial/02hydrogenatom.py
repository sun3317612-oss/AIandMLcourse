"""
3D Hydrogen Atom Hamiltonian Matrix Analysis
수소 원자의 3D 해밀토니안 행렬 분석

3D Schrodinger Equation with Coulomb Potential:
-hbar^2/2m * nabla^2 psi + V(r)psi = E*psi
V(r) = -e^2/r (Coulomb potential)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import os

print("="*80)
print("3D Hydrogen Atom Hamiltonian Matrix Analysis")
print("="*80)

# Physical constants (atomic units: hbar = m = e = 1)
hbar = 1.0
m = 1.0
e = 1.0

# Output directory
output_dir = './'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================================================================
# 3D Hamiltonian Construction
# ============================================================================

def create_hamiltonian_3d(x, y, z, V_3d):
    """
    3D 해밀토니안 행렬 생성 (Finite Difference Method)
    
    Parameters:
    -----------
    x, y, z : 1D arrays
        각 축의 그리드 포인트
    V_3d : 3D array
        포텐셜 에너지 (shape: (Nx, Ny, Nz))
    
    Returns:
    --------
    H : sparse matrix
        해밀토니안 행렬
    T : sparse matrix
        운동 에너지 행렬
    V_flat : 1D array
        포텐셜 에너지 (flattened)
    """
    Nx, Ny, Nz = len(x), len(y), len(z)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    N = Nx * Ny * Nz
    
    print(f"  Building 3D Hamiltonian: {Nx}x{Ny}x{Nz} = {N} points")
    print(f"  Matrix size: {N} x {N} = {N*N:,} elements")
    print(f"  Memory (dense): {N*N*8/1024/1024:.1f} MB")
    
    # 3D 인덱스를 1D 인덱스로 변환하는 함수
    def idx_3d_to_1d(i, j, k):
        return i * Ny * Nz + j * Nz + k
    
    # Sparse 행렬 사용 (메모리 효율)
    T = lil_matrix((N, N))
    
    # 7-point stencil for 3D Laplacian
    coeff_x = hbar**2 / (2 * m * dx**2)
    coeff_y = hbar**2 / (2 * m * dy**2)
    coeff_z = hbar**2 / (2 * m * dz**2)
    coeff_center = 2 * (coeff_x + coeff_y + coeff_z)
    
    print("  Building kinetic energy matrix (7-point stencil)...")
    
    for i in range(Nx):
        if i % 5 == 0:
            print(f"    Progress: {i}/{Nx} ({100*i/Nx:.0f}%)", end='\r')
        for j in range(Ny):
            for k in range(Nz):
                idx = idx_3d_to_1d(i, j, k)
                
                # Center point
                T[idx, idx] = coeff_center
                
                # x-direction neighbors
                if i > 0:
                    idx_neighbor = idx_3d_to_1d(i-1, j, k)
                    T[idx, idx_neighbor] = -coeff_x
                if i < Nx - 1:
                    idx_neighbor = idx_3d_to_1d(i+1, j, k)
                    T[idx, idx_neighbor] = -coeff_x
                
                # y-direction neighbors
                if j > 0:
                    idx_neighbor = idx_3d_to_1d(i, j-1, k)
                    T[idx, idx_neighbor] = -coeff_y
                if j < Ny - 1:
                    idx_neighbor = idx_3d_to_1d(i, j+1, k)
                    T[idx, idx_neighbor] = -coeff_y
                
                # z-direction neighbors
                if k > 0:
                    idx_neighbor = idx_3d_to_1d(i, j, k-1)
                    T[idx, idx_neighbor] = -coeff_z
                if k < Nz - 1:
                    idx_neighbor = idx_3d_to_1d(i, j, k+1)
                    T[idx, idx_neighbor] = -coeff_z
    
    print(f"    Progress: {Nx}/{Nx} (100%)    ")
    
    # Convert to CSR format (efficient for arithmetic)
    T = csr_matrix(T)
    
    # Potential energy (diagonal matrix)
    V_flat = V_3d.flatten()
    
    # Total Hamiltonian
    H = T.copy()
    H.setdiag(H.diagonal() + V_flat)
    
    # Matrix statistics
    nnz = H.nnz
    sparsity = (1 - nnz / (N * N)) * 100
    print(f"  Non-zero elements: {nnz:,} ({100-sparsity:.2f}%)")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Memory (sparse): {nnz*8/1024/1024:.2f} MB")
    
    return H, T, V_flat

def coulomb_potential(x, y, z, epsilon=0.05):
    """
    Coulomb potential V(r) = -e^2/r
    
    Parameters:
    -----------
    x, y, z : 1D arrays
        Grid points
    epsilon : float
        Softening parameter to avoid singularity at r=0
        (작은 값으로 설정: 0.05 Bohr)
    
    Returns:
    --------
    V : 3D array
        Potential energy
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2 + epsilon**2)
    V = -e**2 / r
    
    # Cutoff very negative values to improve numerical stability
    V = np.maximum(V, -50.0)
    
    return V

# ============================================================================
# Grid Comparisons
# ============================================================================

print("\n[Hamiltonian Matrix Size Comparison]")
print("-"*80)

grid_configs = [
    {"size": 20, "range": (-8, 8), "label": "Coarse (20x20x20)"},
    {"size": 25, "range": (-8, 8), "label": "Medium (25x25x25)"},
    {"size": 30, "range": (-8, 8), "label": "Fine (30x30x30)"},
]

hamiltonians = []
grids = []
potentials = []

for config in grid_configs:
    print(f"\n{config['label']}")
    print("-"*80)
    
    n = config["size"]
    x_min, x_max = config["range"]
    
    # 3D grid
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(x_min, x_max, n)
    z = np.linspace(x_min, x_max, n)
    
    dx = x[1] - x[0]
    N = n**3
    
    print(f"  Spatial range: [{x_min}, {x_max}] in each dimension")
    print(f"  Grid spacing: dx = dy = dz = {dx:.3f}")
    print(f"  Grid size: {n} x {n} x {n}")
    print(f"  Total points: {N:,}")
    print(f"  Matrix size: {N:,} x {N:,}")
    
    # Coulomb potential
    V = coulomb_potential(x, y, z, epsilon=0.05)
    
    # Hamiltonian
    H, T, V_flat = create_hamiltonian_3d(x, y, z, V)
    
    hamiltonians.append((H, T, V_flat))
    grids.append((x, y, z))
    potentials.append(V)

# ============================================================================
# Use Fine grid for detailed analysis
# ============================================================================

idx = 2  # Fine grid
x, y, z = grids[idx]
V = potentials[idx]
H, T, V_flat = hamiltonians[idx]

Nx, Ny, Nz = len(x), len(y), len(z)
N = Nx * Ny * Nz

print(f"\n{'='*80}")
print(f"Selected grid for visualization: {grid_configs[idx]['label']}")
print(f"Matrix size: {N} x {N}")
print(f"{'='*80}")

# ============================================================================
# Eigenvalue Problem
# ============================================================================

print("\n[Solving Eigenvalue Problem]")
print("-"*80)

n_states = 10
print(f"Computing lowest {n_states} eigenvalues...")
print("(This may take several minutes for fine grid)")
print("Using ARPACK sparse eigensolver...")

try:
    # Use sparse eigenvalue solver (only finds lowest k eigenvalues)
    # sigma=None means find smallest algebraic eigenvalues
    eigenvalues, eigenvectors = eigsh(H, k=n_states, which='SA', 
                                     maxiter=30000, tol=1e-8)
    
    print(f"\nLowest {n_states} energy eigenvalues (in atomic units):")
    print(f"{'State':<8} {'E (Ha)':<12} {'E (eV)':<12} {'Note':<20}")
    print("-"*55)
    
    # Check for degeneracies
    tol = 0.001
    for i in range(n_states):
        E = eigenvalues[i]
        E_eV = E * 27.211  # Hartree to eV
        
        # Identify state type
        if i == 0:
            note = "1s (ground)"
        elif 1 <= i <= 4:
            note = "n=2 (2s, 2p)"
        elif 5 <= i <= 13:
            note = "n=3 or higher"
        else:
            note = "excited/continuum"
        
        print(f"  {i:<6} {E:10.6f}   {E_eV:10.4f}   {note:<20}")
    
    print("\nDegeneracy analysis:")
    for i in range(n_states - 1):
        if abs(eigenvalues[i+1] - eigenvalues[i]) < tol:
            print(f"  States {i} and {i+1} are degenerate (ΔE = {abs(eigenvalues[i+1]-eigenvalues[i]):.6f} Ha)")
    
    print("\nTheoretical hydrogen energies:")
    print(f"{'n':<8} {'E (Ha)':<12} {'E (eV)':<12} {'Degeneracy':<12}")
    print("-"*50)
    for n in range(1, 6):
        E_theory = -1.0 / (2 * n**2)
        E_eV = E_theory * 27.211
        deg = n**2
        print(f"  {n:<6} {E_theory:10.6f}   {E_eV:10.4f}   {deg}×")
        
except Exception as e:
    print(f"\nError solving eigenvalue problem: {e}")
    print("Matrix might be too large or poorly conditioned.")
    eigenvalues = None
    eigenvectors = None

# ============================================================================
# Visualization
# ============================================================================

print("\n[Creating Visualizations]")
print("-"*80)

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# ============================================================================
# 1. Coulomb Potential (2D slices)
# ============================================================================

ax1 = fig.add_subplot(gs[0, :])

# Three slices: xy-plane, xz-plane, yz-plane at center
mid_x, mid_y, mid_z = Nx//2, Ny//2, Nz//2

# xy-plane (z=0)
extent_xy = [x[0], x[-1], y[0], y[-1]]
V_xy = V[:, :, mid_z]

im1 = ax1.imshow(V_xy.T, extent=extent_xy, origin='lower', 
                 cmap='coolwarm', aspect='auto', vmin=-5, vmax=0)
ax1.set_xlabel('x (Bohr radii)', fontsize=12, fontweight='bold')
ax1.set_ylabel('y (Bohr radii)', fontsize=12, fontweight='bold')
ax1.set_title(f'Coulomb Potential V(r) = -e^2/r (z=0 slice)', 
             fontsize=14, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1, label='V(r) (Hartree)')
ax1.plot(0, 0, 'r*', markersize=15, label='Nucleus')
ax1.legend(fontsize=10)

# Circular contours
r_contours = np.linspace(1, 5, 5)
for r_c in r_contours:
    circle = plt.Circle((0, 0), r_c, fill=False, color='white', 
                       linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.add_patch(circle)

# ============================================================================
# 2. Hamiltonian Matrix Structure (sparse pattern)
# ============================================================================

ax2 = fig.add_subplot(gs[1, 0])

# Sample a small region of the matrix to show structure
sample_size = min(500, N)
H_sample = H[:sample_size, :sample_size].toarray()

im2 = ax2.imshow(H_sample, cmap='RdBu_r', aspect='auto', interpolation='nearest')
ax2.set_xlabel('Column Index', fontsize=10, fontweight='bold')
ax2.set_ylabel('Row Index', fontsize=10, fontweight='bold')
ax2.set_title(f'Hamiltonian Matrix (first {sample_size}x{sample_size})', 
             fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Matrix Element')

# ============================================================================
# 3. Matrix Sparsity Pattern
# ============================================================================

ax3 = fig.add_subplot(gs[1, 1])

# Show non-zero pattern
H_spy = H[:sample_size, :sample_size]
ax3.spy(H_spy, markersize=0.5, color='blue')
ax3.set_xlabel('Column Index', fontsize=10, fontweight='bold')
ax3.set_ylabel('Row Index', fontsize=10, fontweight='bold')
ax3.set_title(f'Sparsity Pattern (7-point stencil)', fontsize=11, fontweight='bold')

# Statistics text
nnz = H.nnz
sparsity = (1 - nnz / (N * N)) * 100
text_str = f'Non-zero: {nnz:,}\nSparsity: {sparsity:.1f}%'
ax3.text(0.95, 0.05, text_str, transform=ax3.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# 4. Potential Energy Distribution
# ============================================================================

ax4 = fig.add_subplot(gs[1, 2])

V_hist = V_flat[V_flat > -20]  # Remove very negative values near r=0
ax4.hist(V_hist, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Potential V(r) (Hartree)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Count', fontsize=10, fontweight='bold')
ax4.set_title('Potential Energy Distribution', fontsize=11, fontweight='bold')
ax4.axvline(np.mean(V_hist), color='red', linestyle='--', 
           label=f'Mean: {np.mean(V_hist):.2f}')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 5. Kinetic Energy Matrix Elements
# ============================================================================

ax5 = fig.add_subplot(gs[2, 0])

# Diagonal vs off-diagonal elements
T_diag = T.diagonal()
ax5.hist(T_diag, bins=30, color='orange', alpha=0.7, label='Diagonal')

# Sample some off-diagonal elements
T_lil = T.tolil()
off_diag_samples = []
for i in range(0, min(1000, N), 10):
    row = T_lil.rows[i]
    data = T_lil.data[i]
    for j, val in zip(row, data):
        if i != j:
            off_diag_samples.append(val)

if off_diag_samples:
    ax5.hist(off_diag_samples, bins=30, color='green', alpha=0.7, label='Off-diagonal')

ax5.set_xlabel('Matrix Element Value', fontsize=10, fontweight='bold')
ax5.set_ylabel('Count', fontsize=10, fontweight='bold')
ax5.set_title('Kinetic Energy Matrix Elements', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 6. Energy Eigenvalue Spectrum
# ============================================================================

ax6 = fig.add_subplot(gs[2, 1])

if eigenvalues is not None:
    n_show = min(len(eigenvalues), 10)
    E_show = eigenvalues[:n_show]
    
    ax6.plot(range(n_show), E_show, 'ro-', markersize=10, linewidth=2, 
            label='Computed')
    
    # Theoretical values
    E_theory = [-1.0/(2*n**2) for n in range(1, n_show+1)]
    ax6.plot(range(n_show), E_theory, 'b^--', markersize=8, linewidth=2, 
            alpha=0.7, label='Theory: E_n=-1/(2n^2)')
    
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Ionization')
    ax6.set_xlabel('State Index', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Energy (Hartree)', fontsize=10, fontweight='bold')
    ax6.set_title(f'Energy Spectrum', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'Eigenvalue computation failed\nor not completed',
            ha='center', va='center', fontsize=11, transform=ax6.transAxes)
    ax6.set_title('Energy Spectrum', fontsize=11, fontweight='bold')

# ============================================================================
# 7. Radial Distribution
# ============================================================================

ax7 = fig.add_subplot(gs[2, 2])

# Create radial bins
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2).flatten()

# Potential as function of r
r_bins = np.linspace(0.5, x.max(), 50)
V_radial = [-e**2 / r for r in r_bins]

ax7.plot(r_bins, V_radial, 'b-', linewidth=2, label='V(r) = -1/r')
ax7.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax7.set_xlabel('r (Bohr radii)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Potential Energy (Hartree)', fontsize=10, fontweight='bold')
ax7.set_title('Coulomb Potential (Radial)', fontsize=11, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_ylim(-5, 1)

plt.suptitle(f'3D Hydrogen Atom Hamiltonian Matrix Analysis ({Nx}x{Ny}x{Nz} grid)', 
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{output_dir}hydrogen_atom_matrix_analysis.png', 
           dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir}hydrogen_atom_matrix_analysis.png")
plt.close()

# ============================================================================
# Wavefunction Visualization
# ============================================================================

if eigenvalues is not None and eigenvectors is not None:
    print("\n[Visualizing Wavefunctions]")
    
    fig2, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    n_states_show = min(8, len(eigenvalues))
    
    for n in range(n_states_show):
        ax = axes[n]
        
        # Get wavefunction
        psi_flat = eigenvectors[:, n]
        psi_3d = psi_flat.reshape((Nx, Ny, Nz))
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi_3d)**2) * (x[1]-x[0])**3)
        psi_3d = psi_3d / norm
        
        # Probability density
        prob_3d = np.abs(psi_3d)**2
        
        # Take z=0 slice
        prob_slice = prob_3d[:, :, mid_z]
        
        # Plot
        extent = [x[0], x[-1], y[0], y[-1]]
        im = ax.imshow(prob_slice.T, extent=extent, origin='lower', 
                      cmap='hot', aspect='auto')
        ax.set_xlabel('x (Bohr)', fontsize=10, fontweight='bold')
        ax.set_ylabel('y (Bohr)', fontsize=10, fontweight='bold')
        ax.set_title(f'State {n}: E = {eigenvalues[n]:.4f} Ha', 
                    fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, label='|psi|^2')
        
        # Mark nucleus
        ax.plot(0, 0, 'w*', markersize=10, markeredgecolor='black', 
               markeredgewidth=0.5)
    
    plt.suptitle(f'Hydrogen Atom Wavefunctions (z=0 slice)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}hydrogen_atom_wavefunctions.png', 
               dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}hydrogen_atom_wavefunctions.png")
    plt.close()

# ============================================================================
# Text File Output
# ============================================================================

print("\n[Generating Text Report]")

with open(f'{output_dir}hydrogen_atom_matrix_structure.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("3D Hydrogen Atom Hamiltonian Matrix Structure Analysis\n")
    f.write("="*80 + "\n\n")
    
    f.write("Physical System:\n")
    f.write("  Hydrogen atom in 3D with Coulomb potential\n")
    f.write("  V(r) = -e^2/r where r = sqrt(x^2 + y^2 + z^2)\n")
    f.write("  Atomic units: hbar = m = e = 1\n\n")
    
    f.write("Grid Configuration:\n")
    f.write(f"  Spatial range: [{x[0]:.2f}, {x[-1]:.2f}] Bohr in each dimension\n")
    f.write(f"  Grid size: {Nx} x {Ny} x {Nz}\n")
    f.write(f"  Grid spacing: dx = dy = dz = {x[1]-x[0]:.4f} Bohr\n")
    f.write(f"  Total grid points: {N:,}\n")
    f.write(f"  Matrix size: {N:,} x {N:,}\n")
    f.write(f"  Matrix elements (dense): {N*N:,}\n")
    f.write(f"  Memory (dense): {N*N*8/1024/1024:.1f} MB\n\n")
    
    f.write("Matrix Structure:\n")
    f.write("  Type: 7-point stencil (sparse)\n")
    f.write("  Center point + 6 neighbors (±x, ±y, ±z)\n")
    nnz = H.nnz
    sparsity = (1 - nnz / (N * N)) * 100
    f.write(f"  Non-zero elements: {nnz:,}\n")
    f.write(f"  Sparsity: {sparsity:.2f}%\n")
    f.write(f"  Memory (sparse): {nnz*8/1024/1024:.2f} MB\n\n")
    
    f.write("Hamiltonian Construction:\n")
    f.write("  H = T + V\n")
    f.write("  T: Kinetic energy (3D Laplacian)\n")
    f.write("  V: Coulomb potential (diagonal)\n\n")
    
    dx = x[1] - x[0]
    coeff = hbar**2 / (2 * m * dx**2)
    f.write("Finite Difference Coefficients:\n")
    f.write(f"  T[i,i] (center): 6 x hbar^2/(2m x dx^2) = {6*coeff:.4f}\n")
    f.write(f"  T[i,i±1] (neighbors): -hbar^2/(2m x dx^2) = {-coeff:.4f}\n\n")
    
    if eigenvalues is not None:
        f.write(f"Energy Eigenvalues (lowest {len(eigenvalues)}):\n")
        f.write(f"{'State':<8} {'E (Ha)':<14} {'E (eV)':<12} {'Degeneracy':<15}\n")
        f.write("-"*55 + "\n")
        
        # Identify degeneracies and quantum numbers
        tol = 0.001  # tolerance for degeneracy
        state_labels = []
        prev_E = None
        deg_count = 0
        
        # Expected degeneracies: n=1(1), n=2(4), n=3(9), n=4(16)
        n_assignment = []
        count = 0
        current_n = 1
        max_states = [1, 4, 9, 16, 25]  # 1s, 2(s+p), 3(s+p+d), etc.
        
        for i in range(len(eigenvalues)):
            E = eigenvalues[i]
            E_eV = E * 27.211
            
            # Check degeneracy
            if prev_E is not None and abs(E - prev_E) < tol:
                deg_count += 1
            else:
                deg_count = 1
            
            # Assign n quantum number
            if count >= max_states[current_n - 1] and current_n < 5:
                current_n += 1
                count = 0
            n_assignment.append(current_n)
            count += 1
            
            # Label
            if current_n == 1:
                label = "1s"
            elif current_n == 2:
                if i == 1:
                    label = "2s/2p"
                else:
                    label = f"2p (deg {deg_count})"
            elif current_n == 3:
                label = f"3s/3p/3d"
            else:
                label = f"n>={current_n}"
            
            f.write(f"  {i:<6} {E:12.6f}   {E_eV:10.4f}   {label:<15}\n")
            prev_E = E
        
        f.write("\n")
        
        f.write("Comparison with Theory:\n")
        f.write(f"{'n':<6} {'Theory (Ha)':<14} {'Computed (Ha)':<16} {'Error %':<10}\n")
        f.write("-"*50 + "\n")
        
        # Match computed states with theoretical n
        comparison = [
            (1, eigenvalues[0]),  # 1s
            (2, eigenvalues[1]),  # 2s or 2p (they're degenerate in hydrogen)
        ]
        
        # Only compare if we have negative energies
        for n, E_comp in comparison:
            E_theory = -1.0 / (2 * n**2)
            error_pct = abs((E_comp - E_theory) / E_theory * 100)
            f.write(f"  {n:<4} {E_theory:12.6f}   {E_comp:14.6f}   {error_pct:8.2f}%\n")
        
        f.write("\n")
        
        f.write("Theoretical Hydrogen Energies (exact):\n")
        f.write(f"{'n':<6} {'E (Ha)':<14} {'E (eV)':<12} {'States':<10}\n")
        f.write("-"*45 + "\n")
        for n in range(1, 6):
            E_theory = -1.0 / (2 * n**2)
            E_eV = E_theory * 27.211
            degeneracy = n**2  # n^2 fold degeneracy in pure Coulomb
            f.write(f"  {n:<4} {E_theory:12.6f}   {E_eV:10.4f}   {degeneracy:<10}\n")
    else:
        f.write("Eigenvalue computation not completed.\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Notes:\n")
    f.write("="*80 + "\n")
    f.write("- The computed energies may differ from theoretical values due to:\n")
    f.write("  1. Finite grid resolution (discretization error)\n")
    f.write("     * Finer grid -> better accuracy\n")
    f.write("     * dx = {:.4f} Bohr is used\n".format(x[1]-x[0]))
    f.write("  2. Softening parameter epsilon = 0.05 Bohr\n")
    f.write("     * Avoids singularity at r=0\n")
    f.write("     * Smaller epsilon -> closer to true Coulomb\n")
    f.write("  3. Finite box size: [-8, 8] Bohr\n")
    f.write("     * Wavefunctions decay but don't reach zero at boundary\n")
    f.write("     * Larger box -> less boundary effect\n")
    f.write("  4. 3D Cartesian grid vs spherical symmetry\n")
    f.write("     * Spherical coordinates would be more natural\n")
    f.write("     * But Cartesian is simpler to implement\n")
    f.write("\n")
    f.write("- Expected accuracy:\n")
    f.write("  * Ground state (1s): ~5-15% error typical for this grid\n")
    f.write("  * Excited states (2s, 2p): ~5-10% error\n")
    f.write("  * Higher states: larger errors due to:\n")
    f.write("    - More extended wavefunctions\n")
    f.write("    - Box boundary effects\n")
    f.write("    - Level mixing in Cartesian grid\n")
    f.write("\n")
    f.write("- Degeneracy in Cartesian vs Spherical:\n")
    f.write("  * Pure Coulomb potential has spherical symmetry\n")
    f.write("  * In spherical coords: n^2-fold degeneracy\n")
    f.write("    (e.g., n=2 has 2s, 2px, 2py, 2pz = 4 states)\n")
    f.write("  * Cartesian grid breaks this symmetry slightly\n")
    f.write("  * Degeneracies should still appear (within ~0.001 Ha)\n")
    f.write("\n")
    f.write("- Why higher states have large apparent errors:\n")
    f.write("  * States 4+ may not correspond to bound hydrogen states\n")
    f.write("  * They might be:\n")
    f.write("    - Box states (particle in finite box)\n")
    f.write("    - Continuum states (E > 0)\n")
    f.write("    - Mixed states due to Cartesian grid\n")
    f.write("  * Only compare states 0-3 with n=1,2 hydrogen levels\n")
    f.write("\n")
    f.write("- Sparse matrix structure:\n")
    f.write("  * 7-point stencil creates 7 diagonals in the matrix\n")
    f.write("  * Band positions: center, +/-1, +/-Nz, +/-(Ny*Nz)\n")
    f.write("  * With Nz={}, Ny*Nz={}\n".format(len(z), len(y)*len(z)))
    f.write("  * This is why you see multiple diagonal bands!\n")
    f.write("\n")
    f.write("- Sparse matrix methods essential for computational efficiency\n")
    f.write("  * Dense: {:.1f} MB\n".format(N*N*8/1024/1024))
    f.write("  * Sparse: {:.2f} MB ({}x reduction!)\n".format(
        nnz*8/1024/1024, int(N*N*8/nnz/8)))

print(f"✓ Saved: {output_dir}hydrogen_atom_matrix_structure.txt")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print("\nGenerated Files:")
print("  1. hydrogen_atom_matrix_analysis.png - Matrix structure visualization")
print("  2. hydrogen_atom_wavefunctions.png - Wavefunction visualization")
print("  3. hydrogen_atom_matrix_structure.txt - Detailed matrix information")
print("\nNote: Computation time depends on grid resolution.")
print("For finer grids (25x25x25 or larger), computation may take several minutes.")

