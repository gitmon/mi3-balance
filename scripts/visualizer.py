import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import polyscope as ps
import gpytoolbox as gp
import numpy as np
import mitsuba as mi
import drjit as dr
from drjit.auto import Float

def plot_sphere(center: np.ndarray = np.zeros(3), radius: float = 1.0, label: str = None):
    V, F = gp.icosphere(n=4)
    V = V * radius + center[None, :]
    return ps.register_surface_mesh(label if label is not None else "Sphere", V, F)

def plot_mesh(V: np.ndarray, F: np.ndarray, label: str = None):
    return ps.register_surface_mesh(label if label is not None else "Mesh", V, F)

def plot_rays(rays: mi.Ray3f, label: str = None):
    o, d = rays.o.numpy().T, rays.d.numpy().T
    return ps.register_point_cloud(label if label is not None else "Rays", o) \
            .add_vector_quantity("dirs", d, enabled=True)