"""
Zeo2D
=====

Plot 3D zeolites as 2D illustrations.

Dependencies
------------
NumPy:      https://numpy.org/
Matplotlib: https://matplotlib.org/
ASE:        https://wiki.fysik.dtu.dk/ase/

"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, Axes
from ase.visualize import view
from ase.io import read as readcif
from ase.build import surface as cleave

def bond_pair(atoms: np.ndarray, bt=(0.2, 2)) -> np.ndarray:
    """Calculate bond under given bond threshold.

    Return paired atoms for plot bonds.
    All bonds are projected to 2D xy plane, therefore may contain duplicated
    (overlayed) lines if calculation is performed in 3D structure.

    Parameters
    ----------
    atoms: array
        array of atom positions (2D or 3D).
    bs: tuple(int, int)
        bond threshold, 
        e.g. (0.2, 2) means distance of atom between 0 and 2 (Å).

    """
    paired = []
    for i, atom in enumerate(atoms):
        dist = norm(atom - atoms[i:], axis=1)
        bonded = atoms[i:][(dist >= bt[0]) & (dist <= bt[1])]
        # Only keep 2D data and discard too "short" bonds.
        paired += [[atom[0:2], bonded_atom[0:2]] for bonded_atom in bonded \
                   if norm(atom[0:2] - bonded_atom[0:2]) >= 0.2]
    return np.array(paired)


def rm_dup_bond(paired: np.ndarray, threshold=0.1) -> list:
    """Remove duplicated bonds."""
    return [pair for i, pair in enumerate(paired) if not np.any(
        [norm(pair - _pair) <= threshold for _pair in paired[i+1:]])]


def rm_dup_atom(atoms: np.ndarray, threshold=0.1) -> np.ndarray:
    """Remove duplicated atoms.

    Atoms could be in 3D or 2D, but 3D atoms will be projected to 2D xy plane.

    """
    atoms = atoms[:, 0:2]
    return np.array([atom for i, atom in enumerate(atoms) if not np.any(
        [norm(atom - _atom) <= threshold for _atom in atoms[i+1:]])])


def plot_balls(atoms: np.ndarray, ax: Axes, bt=(0.2, 2),
               nodup=True, threshold=0.1, **kwargs) -> None:
    """Plot bonds on given Axes.

    Parameters
    ----------
    atoms: array
        Array of atom positions (2D or 3D).
    ax: matplotlib.axes.Axes
        Axes to plot on.
    bs: tuple(int, int)
        Bond threshold, 
        e.g. (0.2, 2) means distance of atom between 0.2 and 2 (Å).
    nodup: bool
        No duplicate: if several bonds are overlayed, only plot one of them.
    threshold: float
        Threshold of two bonds to regard as duplicate.
    **kwargs: dict
        Matplotlib plot kwargs.
        Default style: {'s': 32, 'lw': 2, 'c': 'w', 'edgecolors': 'tan'}.

    """
    style = {'s': 32, 'lw': 2, 'c': 'w', 'edgecolors': 'tan'}
    style.update(kwargs)
    if nodup:
        atoms = rm_dup_atom(atoms)
    ax.scatter(atoms[:, 0], atoms[:, 1], **style)


def plot_sticks(atoms: np.ndarray, ax: Axes, bt=(0.2, 2),
                nodup=True, threshold=0.1, **kwargs) -> None:
    """Plot bonds on given Axes.

    Parameters
    ----------
    atoms: array
        Array of atom positions (2D or 3D).
    ax: matplotlib.axes.Axes
        Axes to plot on.
    bs: tuple(int, int)
        Bond threshold, 
        e.g. (0.2, 2) means distance of atom between 0.2 and 2 (Å).
    nodup: bool
        No duplicate: if several bonds are overlayed, only plot one of them.
    threshold: float
        Threshold of two bonds to regard as duplicate.
    **kwargs: dict
        Matplotlib plot kwargs.
        Default style: {'color': 'tan', 'lw': 2, 'zorder': -1}.

    """
    style = {'color': 'tan', 'lw': 2, 'zorder': -1}
    style.update(kwargs)
    paired = bond_pair(atoms, bt)
    if nodup:
        paired = rm_dup_bond(paired)
    for pair in paired:
        ax.plot(pair[:, 0], pair[:, 1], **style)

def plot_stick_and_ball(atoms: np.ndarray, ax: Axes, bt=(0.2, 2), nodup=True,
                        nostick=False, noball=False,
                        stick={'color': 'tan', 'lw': 2, 'zorder': -1},
                        ball={'s': 32, 'lw': 2, 'c': 'w', 'edgecolors': 'tan'}
                        ) -> None:
    """Wrap plot_sticks and plot_balls."""
    if not nostick:
        plot_sticks(atoms, ax, bt=bt, nodup=nodup, **stick)
    if not noball:
        plot_balls(atoms, ax, nodup=nodup, **ball)

def plot_zeolite(cif: str, surface: (int, int, int),
                 supercell=(2, 2, 1), plot_O=False, nodup=True,
                 Si_style={'s': 32, 'lw': 2, 'c': 'w', 'edgecolors': 'tan'},
                 O_style={'s': 24, 'lw': 0, 'c': 'tan', 'edgecolors': ''},
                 bond_style={'color': 'tan', 'lw': 2, 'zorder': -1}
                 ) -> (Figure, Axes):
    """Plot a zeolite and project all atoms onto xy plane.

    Parameters
    ----------
    cif: str
        File path of *.cif structure (other format may also work).
    surface: tuple
        Miller index of the surface to plot, e.g. (1, 0, 0).
    supercell: tuple
        Scale in x and y direction, do not increase z.
    plot_O: bool
        If False, only plot T atoms (Si) in a zeolite;
        if True, O atoms will also be ploted (not recommended).
    nodup: bool
        Remove overlayed bonds.
    Si_style: dict
        Matplotlib scatter kwargs.
    O_style: dict
        Matplotlib scatter kwargs.
    bond_style: dict
        Matplotlib plot kwargs.
    
    """
    fig, ax = plt.subplots(dpi=300)
    ax.axis('off')
    ax.set_aspect('equal')
    cell = readcif(cif)
    surf = cleave(cell, surface, 1) * supercell
    if plot_O:
        atoms = np.array([atom.position for atom in surf])
        plot_sticks(atoms, ax, bt=(0.2, 2), nodup=nodup, **bond_style)
        O_atoms = np.array([atom.position for atom in surf[[
            atom.index for atom in surf if atom.symbol == 'O']]])
        plot_balls(O_atoms, ax, **O_style)
        Si_atoms = np.array([atom.position for atom in surf[[
            atom.index for atom in surf if atom.symbol == 'Si']]])
        plot_balls(Si_atoms, ax, **Si_style)
        return fig, ax
    del surf[[atom.index for atom in surf if atom.symbol == 'O']]
    atoms = np.array([atom.position for atom in surf])
    plot_stick_and_ball(atoms, ax, bt=(0.2, 3.5), nodup=nodup, 
                        stick=bond_style, ball=Si_style)
    return fig, ax

# Demo
# np.random.seed(seed=10)
# atoms = np.random.rand(64, 2) * 10
# fig, ax = plt.subplots(dpi=300)
# ax.axis('off')
# plot_stick_and_ball(atoms, ax, (0.32, 1.28))