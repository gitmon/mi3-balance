import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import drjit as dr
import mitsuba as mi

import numpy as np
import gpytoolbox as gp
import polyscope as ps

GEO_CENTER = np.zeros(3)
GEO_RADIUS = 1.0
EMITTER_INTENSITY = np.pi
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

def make_textured_geometry(center, radius) -> dict:
    geo_dict = {
        "type": "sphere",
        "center": [center[0], center[1], center[2]],
        "radius": radius,
        "flip_normals": True,
        "bsdf": {
            "type": "diffuse",
            "reflectance": {
                "type": "bitmap",
                # "filename": "/home/jonathan/Documents/mi3-balance/resources/data/common/textures/floor_tiles.jpg"
                # "filename": "/home/jonathan/Documents/mi3-balance/resources/data/common/textures/leaf_mask.png"
                "filename": "/home/jonathan/Documents/mi3-balance/resources/data/common/textures/carrot.png",
                # "filename": "/home/jonathan/Documents/mi3-balance/resources/data/common/textures/noise_02.png"
                # "filename": "/home/jonathan/Documents/mi3-balance/resources/data/common/textures/museum.exr"
                "filter_type": "nearest",
                "raw": True
            }
        }
    }
    return geo_dict

def make_emitter(center, emitter_intensity) -> dict:
    emitter_dict = {
        "type": "point",
        "position": [center[0], center[1], center[2] + 0.4],
        "intensity": {
            "type": "uniform",
            "value": emitter_intensity,
        }
    }
    return emitter_dict

def make_sensor(spp: int = 32) -> dict:
    camera = {
        "type": "perspective",
        "fov": 135,
        "to_world": mi.ScalarTransform4f().look_at(
            origin=mi.ScalarPoint3f([0.0, 0.0, 0.0]),
            target=mi.ScalarPoint3f([1.0, 0.0, 0.0]),
            up=    mi.ScalarPoint3f([0.0, 0.0, 1.0])),
        "film": {
            "type": "hdrfilm",
            "width": 256,
            "height": 256,
        },
        "sampler": {
            "type": "independent",
            "sample_count": spp,
        }
    }
    return camera


def make_scene(color_ref = BSDF_COLOR_REF, use_texture = False):
    scene_dict = {
        "type": "scene",
        "myintegrator": {
            "type": "path",
            "max_depth": 10 # 5
        },
        "mysphere": make_textured_geometry(GEO_CENTER, GEO_RADIUS) if use_texture else make_geometry(GEO_CENTER, GEO_RADIUS, color_ref),
        "myemitter": make_emitter(GEO_CENTER, EMITTER_INTENSITY),
        "mysensor": make_sensor(SENSOR_SPP),
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