# %%[markdown]
# # Zeo2D
# Utilities for plotting zeolite illustrations.
# %%
from zeo2d import readcif, view, cleave, plot_stick_and_ball, plot_zeolite
import matplotlib.pyplot as plt
# %%[markdown]
# ## Easy plot
# Plot ball and stick.
# %%
plot_zeolite("iza-structure.org/BEA.cif", surface=(1, 0, 0), supercell=(4, 2, 1))
# %%[markdown]
# Or plot just stick.
# %%
plot_zeolite("iza-structure.org/LTA.cif", surface=(0, 0, 1),
    Si_style={'s': 0, 'lw': 0})
# %%[markdown]
# ## Plot with O atoms.
# %%
plot_zeolite("iza-structure.org/LTA.cif", (0, 0, 1), plot_O=True,
    O_style={'s': 24, 'lw': 0, 'c': 'wheat', 'edgecolors': ''},
    Si_style={'s': 48, 'lw': 0, 'c': 'tan', 'edgecolors': ''},
    bond_style={'color': 'wheat', 'lw': 2, 'zorder': -1})
# %%[markdown]
# ## Plot with some modifications
# Here demonstrates how to create an array of atoms postions for plotting.
# %%
cell = readcif("iza-structure.org/LTL.cif")
surf = cleave(cell, (0, 0, 1), 1) * (2, 2, 1)
surf.rotate(120, 'z', center='COU')
# Preview of 3D structure, may not rendering properly.
view(surf, viewer='x3d')
# %%
atoms = surf[[atom.index for atom in surf if atom.symbol == 'Si']].positions
fig, ax = plt.subplots(dpi=300, )
fig.set_size_inches(20 / 2.54, 20 / 2.54)
ax.axis('off')
ax.set_aspect('equal')
plot_stick_and_ball(atoms, ax, bt=(0.2, 3.5), 
    stick={'color': 'lightskyblue', 'lw': 6},
    ball={'s': 100, 'lw': 4, 'c': 'w', 'edgecolors': 'plum'})

# %%[markdown]
# ## Save an illustration
# %%
fig, ax = plot_zeolite("iza-structure.org/AFI.cif",
    surface=(0, 0, 1), supercell=(4, 4, 1))
fig.savefig('AFI.svg')
