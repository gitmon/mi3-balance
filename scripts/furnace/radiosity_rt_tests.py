import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import numpy as np
import mitsuba as mi
import drjit as dr
from radiosity_rt import SceneSurfaceSampler, RadianceCacheMITSUBA
from furnace_scene import make_geometry, make_emitter, make_sensor

def furnace_scene_nonemissive():
    '''
    Unit radius
    Point light intensity = pi
    Albedo = [0.5, 0.5, 0.5]
    '''
    center = np.zeros(3)
    radius = 1.0
    emitter_I = np.pi
    albedo_scalar = 0.5
    albedo = np.full((3), albedo_scalar)
    scene_dict = {
        "type": "scene",
        "myintegrator": {
            "type": "path",
            "max_depth": 10
        },
        "mysphere": make_geometry(center, radius, albedo),
        "myemitter": make_emitter(center, emitter_I),
        "mysensor": make_sensor(),
    }
    scene = mi.load_dict(scene_dict)

    sampler_rt = mi.load_dict({'type': 'independent'})
    scene_sampler = SceneSurfaceSampler(scene)

    num_points = 1024
    num_wi = 256
    si = scene_sampler.sample(num_points, sampler_rt)[0]
    radiance_cache = RadianceCacheMITSUBA(scene, 1024, 1024)
    Lo = radiance_cache.query_cached_Lo(si, sampler_rt)
    Lo_err = np.mean((Lo - 1.0).numpy(), axis=0)    # should be 1.0

    Li, wi_local = radiance_cache.query_cached_Li(si, num_wi, sampler_rt)[:2]
    integrand = Li * albedo_scalar * dr.inv_pi * mi.Frame3f.cos_theta(wi_local)
    rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
    Li_err = np.mean((rhs - 1.0).numpy(), axis=0)   # should be 1.0(??)

    return Lo_err, Li_err

def furnace_scene_emissive(radius: float = 1.0):
    '''
    Sphere emissivity = 0.5
    Radius is arbitrary
    Albedo = [0.5, 0.5, 0.5]
    '''
    center = np.zeros(3)
    sphere_I = 0.5
    albedo_scalar = 0.5
    albedo = np.full((3), albedo_scalar)
    
    sph_dict = make_geometry(center, radius, albedo)
    sph_dict['emitter'] = {
        'type': 'area',
        'radiance': {
            'type': 'rgb',
            'value': sphere_I,
        }
    }
    scene_dict = {
        "type": "scene",
        "myintegrator": {
            "type": "path",
            "max_depth": 20
        },
        "mysphere": sph_dict,
        "mysensor": make_sensor(),
    }
    scene = mi.load_dict(scene_dict)

    sampler_rt = mi.load_dict({'type': 'independent'})
    scene_sampler = SceneSurfaceSampler(scene)

    num_points = 1024
    num_wi = 256
    si = scene_sampler.sample(num_points, sampler_rt)[0]
    radiance_cache = RadianceCacheMITSUBA(scene, 1024, 1024)
    Lo = radiance_cache.query_cached_Lo(si, sampler_rt)
    Lo_err = np.mean((Lo - 1.0).numpy(), axis=0)    # should be 1.0

    Li, wi_local = radiance_cache.query_cached_Li(si, num_wi, sampler_rt)[:2]  # should be 1.0
    integrand = Li * albedo_scalar * dr.inv_pi * mi.Frame3f.cos_theta(wi_local)
    rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi # should be 0.5?
    Li_err = np.mean((rhs - 0.5).numpy(), axis=0)

    return Lo_err, Li_err
