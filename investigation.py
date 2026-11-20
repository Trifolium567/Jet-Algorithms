# %%
import data_handling
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
import four_vectors as fv

event_list = data_handling.events_from_file("data/jets.dat")

# %%

sample1 = event_list[0].make_cuts(True)
clustered_indices, clustered_four_vectors = sample1.kt_algorithm(-1, 1, 2)  # ty: ignore[missing-union-attribute]

# %%

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
fig2, ax2 = plt.subplots()

colours = cmaps['tab20'].colors  # ty: ignore[missing-attribute]

unclustered = []
for i, cluster in enumerate(clustered_indices):
    if len(cluster) == 1:
        unclustered += cluster
    else:
        cluster_four_vectors = sample1.get_four_vectors(cluster)  # ty: ignore[missing-union-attribute]

        y = cluster_four_vectors.rapidity_array
        phi = cluster_four_vectors.azimuthal_angle_array
        p_T = cluster_four_vectors.p_T_array

        width = depth = 0.1
        bottom = np.zeros_like(p_T)

        ax.bar3d(y, phi, bottom, width, depth, p_T, shade=True,  # label=i,
                 color=colours[i])
        ax2.scatter(y, phi, color=colours[i],)  # label=i, )


unclustered_four_vectors = sample1.get_four_vectors(unclustered)  # ty: ignore[missing-union-attribute]

y = unclustered_four_vectors.rapidity_array
phi = unclustered_four_vectors.azimuthal_angle_array
p_T = unclustered_four_vectors.p_T_array

# width = depth = 0.1
top = unclustered_four_vectors.p_T_array
bottom = np.zeros_like(top)
ax.bar3d(y, phi, bottom, width, depth, p_T, shade=True, color="grey",
         label="unclustered")
ax2.scatter(y, phi, color="grey", label="unclustered")

ax.set_title("Clustered Particles, R=1, k=-1")
ax.set_zlabel("p_T")
ax.set_ylabel("azimuthal angle")
ax.set_xlabel("rapidity")

ax2.set_title("Clustered Particles, R=1, k=-1")
ax2.set_xlabel("rapidity")
ax2.set_ylabel("azimuthal angle")
ax.legend()
ax2.legend()
plt.show()

# %%