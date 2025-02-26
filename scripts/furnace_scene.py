import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import drjit as dr
import mitsuba as mi

import numpy as np
import gpytoolbox as gp
import polyscope as ps

GEO_CENTER = np.zeros(3)
GEO_RADIUS = 1.0
EMITTER_INTENSITY = 1.0
BSDF_COLOR_REF = [0.2, 0.25, 0.7]
SENSOR_COUNT = 1
SENSOR_SPP = 64

def make_geometry(center, radius, color_ref) -> dict:
    geo_dict = {
        "type": "sphere",
        "center": [center[0], center[1], center[2]],
        "radius": radius,
        "flip_normals": True,
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "rgb",
                "value": color_ref
            }
        }
    }
    return geo_dict

def make_emitter(center, emitter_intensity) -> dict:
    emitter_dict = {
        "type": "point",
        "position": [center[0], center[1], center[2] + 0.1],
        "intensity": {
            "type": "uniform",
            "value": emitter_intensity,
        }
    }
    return emitter_dict

def make_sensor(origins: np.ndarray, directions: np.ndarray, spp: int) -> dict:
    probes = {
        "type": "batch",
            "film": {
                "type": "hdrfilm",
                "rfilter": { "type": "box" },
                "width": origins.shape[0],
                "height": 1,
            },
            "sampler": {
                "type": "independent",
                "sample_count": spp,
            }
    }

    for idx, (origin, direction) in enumerate(zip(origins, directions)):
        sensor_name = f"sensor{idx}"
        probe = {
            "type": "radiancemeter",
            "origin": [origin[0], origin[1], origin[2]],
            "direction": [direction[0], direction[1], direction[2]],
            "film": {
                "type": "hdrfilm",
                "pixel_format": "rgb",
                "rfilter": { "type": "box" },
                "width": 1,
                "height": 1,
            },
        }
        probes[sensor_name] = probe

    return probes

def make_scene(color_ref = BSDF_COLOR_REF):
    ts = np.linspace(0, 2 * np.pi, SENSOR_COUNT)
    origins = np.c_[np.cos(ts), np.sin(ts), np.zeros_like(ts)] * GEO_RADIUS
    directions = GEO_CENTER[None,:] - origins
    directions /= np.linalg.norm(directions, axis=1)[:,None]

    scene_dict = {
        "type": "scene",
        "myintegrator": {
            "type": "path",
            "max_depth": 5
        },
        "mysphere": make_geometry(GEO_CENTER, GEO_RADIUS, color_ref),
        "myemitter": make_emitter(GEO_CENTER, EMITTER_INTENSITY),
        "mysensor": make_sensor(origins, directions, SENSOR_SPP),
    }
    scene = mi.load_dict(scene_dict)
    return scene

def visualize_scene(sph_center, sph_radius, emitter_center, probe_origins, probe_directions):
    V, F = gp.icosphere(n=4)

    ps.init()

    # show geo
    ps.register_surface_mesh("Sphere", V * sph_radius + sph_center[None,:], F)

    # show emitters
    # ps.register_surface_mesh("Emitter", V * 0.01 + emitter_center[None,:], F, color=[1,1,0])
    ps.register_point_cloud("Emitter", emitter_center[None,:], color=[1,1,0])

    # show probes
    probes = ps.register_point_cloud("Probes", probe_origins)
    probes.add_vector_quantity("LookAt", probe_directions)
    ps.show()

if __name__ == "__main__":
    make_scene()