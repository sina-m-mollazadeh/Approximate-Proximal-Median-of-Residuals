import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Set random seed for reproducibility (identical to previous code)
np.random.seed(7)

# 1. Define a non-linear 2D regression surface function
def regression_surface(x1, x2):
    return (3.5 * np.sin(0.65 * x1) * np.cos(0.65 * x2) 
            + 0.15 * (x1**2 - x2**2) 
            + 0.4 * x1 - 0.3 * x2)

# 2. Generate 5000 random samples for the 2D x vector in [-5, 5] x [-5, 5]
N = 5000
x1_all = np.random.uniform(-5, 5, N)
x2_all = np.random.uniform(-5, 5, N)

# Generate the response vector b around the true regression surface
true_b_all = regression_surface(x1_all, x2_all)
b_all = true_b_all + np.random.normal(0, 2.5, N)

# 3. Choose 10 random x tuples in [-5, 5] x [-5, 5]
num_tuples = 10
x1_tuples = np.random.uniform(-4.5, 4.5, num_tuples)
x2_tuples = np.random.uniform(-4.5, 4.5, num_tuples)

# Number of predicted sample values per x tuple
n_samples_per_tuple = 50

# Define the median order statistic using ceil(n/2) (1-based) -> ceil(n/2) - 1 (0-based)
# This guarantees the median is strictly one of the sample indices (or two if tied).
k_median = int(np.ceil(n_samples_per_tuple / 2.0)) - 1

# Choose one of the 10 randomly selected x vectors (Tuple #5 at (-0.18, -1.07))
selected_tuple_id = 4
x_sel = np.array([x1_tuples[selected_tuple_id], x2_tuples[selected_tuple_id]])

# 4. Define continuous sample functions across the (x1, x2) space so each sample i in [0, n-1]
# has a well-defined residual and we can track the exact sample index at rank ceil(n/2).
rng = np.random.default_rng(77)
base_offsets = np.sort(rng.normal(0, 2.3, n_samples_per_tuple))
base_offsets -= base_offsets[k_median]  # Center the ceil(n/2) sample on the surface at x_sel
base_offsets[:k_median] -= 0.35
base_offsets[k_median + 1:] += 0.35
base_offsets[0] -= 10.5
base_offsets[-1] += 11.0

slopes_x1 = rng.normal(0, 0.30, n_samples_per_tuple)
slopes_x2 = rng.normal(0, 0.30, n_samples_per_tuple)
curv = rng.uniform(0.05, 0.13, n_samples_per_tuple) * np.where(
    np.arange(n_samples_per_tuple) <= k_median, 1.0, -1.0
)
slopes_x1[k_median] = 0.05
slopes_x2[k_median] = -0.05
curv[k_median] = 0.085

def evaluate_samples_at_x(x1, x2):
    x1_arr = np.asarray(x1)[..., None]
    x2_arr = np.asarray(x2)[..., None]
    surf_val = regression_surface(x1_arr, x2_arr)
    dx1 = x1_arr - x_sel[0]
    dx2 = x2_arr - x_sel[1]
    sample_residuals = (base_offsets 
                        + dx1 * slopes_x1 
                        + dx2 * slopes_x2 
                        + (dx1**2 + dx2**2) * curv)
    return surf_val + sample_residuals

def get_ceil_median_indice(x1, x2, tie_tol=0.05):
    """
    Returns:
      - preds: all n sample predictions at (x1, x2)
      - med_indice: the sample index at order statistic ceil(n/2)
      - med_val: the sample value at med_indice (strictly one of the samples)
      - tied_indices: list of sample indices if two samples have nearly the exact same residual
    """
    preds = evaluate_samples_at_x(x1, x2)
    sorted_indices = np.argsort(preds, axis=-1)
    med_indice = sorted_indices[..., k_median]
    med_val = np.take_along_axis(preds, med_indice[..., None], axis=-1).squeeze(-1)
    
    if preds.ndim == 1:
        tied = [int(med_indice)]
        for adj_rank in [k_median - 1, k_median + 1]:
            if 0 <= adj_rank < n_samples_per_tuple:
                adj_idx = int(sorted_indices[adj_rank])
                if abs(preds[adj_idx] - med_val) <= tie_tol:
                    tied.append(adj_idx)
        return preds, int(med_indice), float(med_val), tied
    return preds, med_indice, med_val

# Z-level for the base plane underneath the regression surface
z_base = -20.0

# Evaluate the ceil(n/2) median sample index across a fine grid on the base plane
grid_res = 300
dense_x1, dense_x2 = np.meshgrid(np.linspace(-5, 5, grid_res), np.linspace(-5, 5, grid_res))
_, dense_med_indice, _ = get_ceil_median_indice(dense_x1, dense_x2)

# Find the active median sample index for the chosen x vector (Tuple #5)
_, target_med_indice, target_med_val, target_tied = get_ceil_median_indice(x_sel[0], x_sel[1])
invariant_indice_mask = (dense_med_indice == target_med_indice)

# 5. Initialize the 3D Figure
fig = plt.figure(figsize=(15, 10.5))
ax = fig.add_subplot(111, projection='3d')
ax.computed_zorder = False

# Meshgrid for the base plane and the regression surface
grid_x1, grid_x2 = np.meshgrid(np.linspace(-5, 5, 65), np.linspace(-5, 5, 65))
grid_b = regression_surface(grid_x1, grid_x2)

# Layer 1: Base plane where the random 2D x vectors reside
ax.plot_surface(grid_x1, grid_x2, np.full_like(grid_x1, z_base), 
                color='slategray', alpha=0.14, edgecolor='none', zorder=1)
ax.scatter(x1_all, x2_all, np.full_like(x1_all, z_base), 
           color='dimgray', alpha=0.07, s=2, zorder=1)

# Layer 2: Plot the area underneath the regression surface (on the base plane)
# where the ceil(n/2) median INDICE does not change for the selected x tuple
ax.scatter(dense_x1[invariant_indice_mask], 
           dense_x2[invariant_indice_mask], 
           np.full(np.sum(invariant_indice_mask), z_base),
           color='gold', alpha=0.60, s=7, zorder=2)

# Plot the exact boundary contour of this constant-index area on the base plane
# (Along this boundary, two samples have the exact same residual and tie at rank ceil(n/2))
ax.contour(dense_x1, dense_x2, invariant_indice_mask.astype(float), 
           levels=[0.5], zdir='z', offset=z_base, 
           colors=['orangered'], linewidths=2.8, zorder=3)

# Layer 3: Faint 3D cloud of the 5000 noisy response samples
ax.scatter(x1_all, x2_all, b_all, 
           color='steelblue', alpha=0.04, s=3, zorder=3)

# Layer 4: The actual non-linear 3D regression surface
ax.plot_surface(grid_x1, grid_x2, grid_b, 
                cmap='viridis', alpha=0.50, edgecolor='teal', linewidth=0.15, 
                antialiased=True, zorder=4)

# Layer 5+: Plot all 10 random x tuples, their perpendicular lines, and ceil(n/2) median samples
for i in range(num_tuples):
    xk1, xk2 = x1_tuples[i], x2_tuples[i]
    true_val = regression_surface(xk1, xk2)
    predicted_samples, med_idx, median_val, tied_list = get_ceil_median_indice(xk1, xk2)
    is_selected = (i == selected_tuple_id)
    
    # Plot the random x tuple on the base plane
    if is_selected:
        ax.scatter([xk1], [xk2], [z_base], 
                   color='orangered', marker='D', s=130, edgecolor='black', linewidth=1.5, zorder=6)
    else:
        ax.scatter([xk1], [xk2], [z_base], 
                   color='black', marker='X', s=65, zorder=5)
    
    # Perpendicular vertical line from the base plane through all predicted values
    line_z_max = max(np.max(predicted_samples), true_val) + 1.5
    ax.plot([xk1, xk1], [xk2, xk2], [z_base, line_z_max], 
            color='orangered' if is_selected else 'black', 
            linestyle='-' if is_selected else '--', 
            linewidth=2.4 if is_selected else 1.3, 
            alpha=0.95 if is_selected else 0.75, zorder=5)
    
    # Plot all predicted values along the perpendicular line
    ax.scatter([xk1] * n_samples_per_tuple, [xk2] * n_samples_per_tuple, predicted_samples, 
               color='crimson', s=24, alpha=0.8, zorder=6)
    
    # Plot the true regression surface value at x
    ax.scatter([xk1], [xk2], [true_val], 
               color='dodgerblue', s=90, marker='o', edgecolor='white', linewidth=1.2, zorder=7)
    
    # Plot the ceil(n/2) median sample right on the perpendicular line
    ax.scatter([xk1], [xk2], [median_val], 
               color='gold' if is_selected else 'lime', 
               s=230 if is_selected else 165, 
               marker='*', edgecolor='black', linewidth=1.1, zorder=8)

# Annotate the constant median-index area on the base plane
ax.text(x_sel[0] + 0.95, x_sel[1] - 2.1, z_base, 
        f"Constant Median-Indice Area\n(Indice = Sample #{target_med_indice})", 
        color='darkred', fontsize=9.5, fontweight='bold', zorder=9)

# 6. Formatting and camera angle
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(z_base, 20)
ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)
ax.set_zlabel('Response / Predicted Value ($b$)', fontsize=12, labelpad=10)
ax.set_title('3D Regression Surface with $\\lceil n/2 \\rceil$ Median Samples & Constant Median-Indice Area on Base Plane', 
             fontsize=13, pad=15)

ax.view_init(elev=24, azim=-48)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='5000 Random $x$ Vectors (Base Plane)', markerfacecolor='dimgray', markersize=5, alpha=0.6),
    Line2D([0], [0], marker='o', color='w', label='5000 Noisy Response Samples ($b$)', markerfacecolor='steelblue', markersize=5, alpha=0.5),
    Line2D([0], [0], marker='X', color='w', label='Random $x$ Tuples (on Base Plane)', markerfacecolor='black', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='D', color='w', label=f'Chosen $x$ Vector: Tuple #{selected_tuple_id+1} $({x_sel[0]:.2f}, {x_sel[1]:.2f})$', markerfacecolor='orangered', markeredgecolor='black', markersize=9),
    Patch(facecolor='gold', edgecolor='orangered', linewidth=1.8, alpha=0.75, label=f'Area Where Median Indice Does Not Change (Indice = #{target_med_indice})'),
    Line2D([0], [0], color='orangered', lw=2.2, ls='-', label='Tie Boundary (2 Samples Have Same Residual / Swap Rank)'),
    Line2D([0], [0], color='black', lw=1.4, ls='--', label='Perpendicular Line at Tuple $x$'),
    Line2D([0], [0], marker='o', color='w', label='Predicted Sample Values along Line', markerfacecolor='crimson', markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Actual Regression Surface Value', markerfacecolor='dodgerblue', markeredgecolor='white', markersize=9),
    Line2D([0], [0], marker='*', color='w', label='$\\lceil n/2 \\rceil$ Median Sample (Strictly One of the Samples)', markerfacecolor='lime', markeredgecolor='black', markersize=13),
    Line2D([0], [0], marker='*', color='w', label=f'Chosen Tuple $\\lceil n/2 \\rceil$ Median (Sample #{target_med_indice})', markerfacecolor='gold', markeredgecolor='black', markersize=14),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.98), fontsize=8.6, framealpha=0.93)

plt.tight_layout()
plt.savefig('nonlinear_regression_constant_median_indice_v2.png', dpi=300)
plt.show()