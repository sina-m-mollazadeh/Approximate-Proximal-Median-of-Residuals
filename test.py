import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load data
# ============================================================

A = np.loadtxt("A.csv")

b = np.loadtxt("b.csv")

x_true = np.loadtxt("x.csv")


# ensure b shape is (n,)
b = b.reshape(-1)

print("A shape:", A.shape)
print("b shape:", b.shape)
print("true x:", x_true)


# Load real APMR trajectory
apmr_path = np.loadtxt(
    "apmr_trajectory.csv",
    delimiter=","
)

print("APMR trajectory shape:", apmr_path.shape)



# ============================================================
# Objective
# ============================================================

def median_residual(x):

    r = np.abs(A @ x - b)
    med=np.median(r)
    mini=np.argmin(np.abs(r-med))
    return r[mini]


# ============================================================
# LS trajectory
# ============================================================

def ls_trajectory(
        x0,
        lr=0.01,
        steps=100
):

    x = x0.copy()

    path = [x.copy()]


    n = len(b)
    for _ in range(steps):
        gradient = (
            2 / n
            *
            A.T @ (A @ x - b)
        )


        x = x - lr * gradient

        path.append(x.copy())


    return np.array(path)



# ============================================================
# LAD trajectory
# ============================================================

def lad_trajectory(
        x0,
        lr=0.01,
        steps=100
):

    x = x0.copy()

    path = [x.copy()]


    for _ in range(steps):

        residuals = A @ x - b


        subgradient = (
            A.T @ np.sign(residuals)
        )


        x = x - lr * subgradient


        path.append(x.copy())


    return np.array(path)



x0 = rng.standard_normal(A.shape[1])
x0=np.array([-6.517911526116896281e-01,-1.747172923257771515e-01])


ls_path = ls_trajectory(
    x0,
    lr=0.01,
    steps=len(apmr_path)-1
)


lad_path = lad_trajectory(
    x0,
    lr=0.005,
    steps=len(apmr_path)-1
)



print("LS final:", ls_path[-1])
print("LAD final:", lad_path[-1])
print("APMR final:", apmr_path[-1])



# ============================================================
# Create MAR objective landscape
# ============================================================

# use range based on all trajectories

all_points = np.vstack(
    [
        ls_path,
        lad_path,
        apmr_path,
        x_true.reshape(1,-1)
    ]
)


xmin = all_points[:,0].min() - 1
xmax = all_points[:,0].max() + 1

ymin = all_points[:,1].min() - 1
ymax = all_points[:,1].max() + 1



x1 = np.linspace(
    xmin,
    xmax,
    250
)

x2 = np.linspace(
    ymin,
    ymax,
    250
)


X1, X2 = np.meshgrid(
    x1,
    x2
)


Z = np.zeros_like(X1)


for i in range(X1.shape[0]):

    for j in range(X1.shape[1]):

        point = np.array(
            [
                X1[i,j],
                X2[i,j]
            ]
        )

        Z[i,j] = median_residual(point)



# ============================================================
# Plot
# ============================================================
def draw_path(path, color, marker, label,arrow_color):

    # trajectory
    plt.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=2,
        marker=marker,
        markersize=4,
        label=label,
    )

    # arrows
    for i in range(len(path) - 1):

        plt.annotate(
            "",
            xy=path[i + 1],
            xytext=path[i],
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=2,
                shrinkA=1,
                shrinkB=1,
            ),
        )

plt.figure(
    figsize=(9,7)
)

cf = plt.contourf(
    X1,
    X2,
    Z,
    levels=80,
    cmap="viridis",      # blue -> green
)

contours = plt.contour(
    X1,
    X2,
    Z,
    levels=20,
    colors="k",
    linewidths=0.35,
    alpha=0.35,
)

plt.clabel(
    contours,
    fontsize=7,
    colors="gray"
)
cbar = plt.colorbar(cf)
cbar.set_label("Median Absolute Residual")



# LS path
draw_path(
    ls_path,
    color="#000047",
    marker="o",
    label="LS trajectory",
    arrow_color="#0000A3"
)

draw_path(
    lad_path,
    color="#8B4000",
    marker="s",
    label="LAD trajectory",
    arrow_color="#D16200"
)

draw_path(
    apmr_path,
    color="#023020",
    marker="^",
    label="APMR trajectory",
    arrow_color="#079D68"
)



# final points

plt.scatter(
    ls_path[-1,0],
    ls_path[-1,1],
    s=160,
    color="tab:blue",
    edgecolor="black",
    zorder=5,
)

plt.scatter(
    lad_path[-1,0],
    lad_path[-1,1],
    s=160,
    color="tab:orange",
    edgecolor="black",
    marker="s",
    zorder=5,
)

plt.scatter(
    apmr_path[-1,0],
    apmr_path[-1,1],
    s=240,
    color="tab:green",
    edgecolor="black",
    marker="*",
    zorder=6,
)

plt.scatter(
    x_true[0],
    x_true[1],
    s=220,
    color="red",
    edgecolor="black",
    marker="X",
    label="True parameter",
    zorder=7,
)


plt.xlabel("$x_1$")

plt.ylabel("$x_2$")


plt.title(
    "Optimization trajectories on the MAR objective landscape"
)


plt.legend()

plt.grid(alpha=0.3)


plt.tight_layout()


plt.savefig(
    "APMR_LS_LAD_real_geometry.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()