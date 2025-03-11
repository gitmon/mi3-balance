import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gpytoolbox import per_vertex_normals

from sh_utils import eval_basis, get_sh_count

def fit_sh_on_mesh_vertex(scene: mi.Scene, mesh_idx: int, mesh_vtx_idx: int, max_order: int, Nquad: int, spp: int = 64):
    # Load data from scene
    integrator = scene.integrator()
    meshes = [shape for shape in scene.shapes() if shape.is_mesh()]
    mesh = meshes[mesh_idx]

    # Initialize interaction structs at each of the mesh vertices
    positions = dr.gather(mi.Point3f, dr.unravel(mi.Point3f, mesh.vertex_positions_buffer()), mesh_vtx_idx)
    normals   = dr.gather(mi.Vector3f, dr.unravel(mi.Vector3f, mesh.vertex_normals_buffer()), mesh_vtx_idx)
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.p, si.n = positions, normals
    si.sh_frame = mi.Frame3f(normals)

    # evaluate SH basis
    d, sh_basis, quad_W = eval_basis(max_order, Nquad)
    # d, sh_basis, quad_W = eval_basis_on_hemisphere(max_order, Nquad)
    dirs_per_point = dr.width(d)

    total_rays = dirs_per_point * spp
    ray_batch_size = total_rays

    # Trace and aggregate rays
    si_wide = dr.gather(type(si), si, dr.ones(UInt, spp * dirs_per_point))
    d_wide = dr.gather(type(d), d, dr.repeat(dr.arange(UInt, dirs_per_point), spp))
    si_wide.wi = d_wide

    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, ray_batch_size)
    # measure radiance from position `d` towards `si.p`
    rays = si_wide.spawn_ray(si_wide.to_world(d_wide))
    # rays.o += rays.d * 1.0 * mi.math.RayEpsilon   # added
    rays.d = -rays.d
    colors_Lo, _, _ = integrator.sample(scene, sampler, rays)
    Lo_per_d = dr.block_reduce(dr.ReduceOp.Add, colors_Lo, spp) / spp

    idx2 = dr.arange(UInt, dirs_per_point)
    quadW_wide = dr.gather(type(quad_W), quad_W, idx2)
    sh_basis_wide = dr.gather(type(sh_basis), sh_basis, idx2)

    sh_color = dr.zeros(mi.Color3f, dr.width(Lo_per_d))
    for sh_id, basis_wide in enumerate(sh_basis_wide):
        Lo_coeff = dr.block_reduce(dr.ReduceOp.Add, quadW_wide * basis_wide * Lo_per_d, block_size = dirs_per_point)
        sh_color += Lo_coeff * basis_wide
        mesh.add_attribute(
            name=f"vertex_Lo_coeffs_{sh_id}",
            size=3,
            buffer=dr.ravel(Lo_coeff))
    sh_color = dr.clip(sh_color, 0.0, 1.0)

    plt.figure(figsize=(12,7))
    plt.subplot(211); plt.title("Reference")
    plt.imshow(mi.TensorXf(Lo_per_d).numpy().reshape(3,Nquad+1, -1).swapaxes(0,2))
    plt.subplot(212); plt.title(f"Spherical harmonics fit (degree = {max_order})")
    plt.imshow(mi.TensorXf(sh_color).numpy().reshape(3,Nquad+1, -1).swapaxes(0,2))
    plt.tight_layout()


def fit_sh_on_mesh_unbatched(integrator: mi.Integrator, mesh: mi.Mesh, max_order: int, Nquad: int = 64, spp: int = 32, seed: int = 0):
    Nv = mesh.vertex_count()
    # Initialize interaction structs at each of the mesh vertices
    positions = dr.unravel(mi.Point3f, mesh.vertex_positions_buffer())
    normals = dr.unravel(mi.Vector3f, mesh.vertex_normals_buffer())
    # positions += mi.math.RayEpsilon * normals
    si = dr.zeros(mi.SurfaceInteraction3f, Nv)
    si.p, si.n = positions, normals
    si.sh_frame = mi.Frame3f(normals)

    # evaluate SH basis
    d, sh_basis, quad_W = eval_basis(max_order, Nquad)
    dirs_per_point = dr.width(d)

    # Split computation into batches if needed?
    MAX_RAYS_PER_BATCH = 1 << 29
    total_rays = Nv * dirs_per_point * spp
    print(f"Nv = {Nv}, Nrays = {total_rays}")
    ray_batch_size = total_rays
    num_batches = (total_rays + MAX_RAYS_PER_BATCH) // MAX_RAYS_PER_BATCH
    if num_batches > 1:
        raise NotImplementedError()
    
    # Trace and aggregate rays
    si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, Nv), spp * dirs_per_point))
    d_wide = dr.gather(type(d), d, dr.tile(dr.repeat(dr.arange(UInt, dirs_per_point), spp), Nv))
    si_wide.wi = d_wide

    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(seed, ray_batch_size)
    # measure radiance from position `d` towards `si.p`
    rays = si_wide.spawn_ray(si_wide.to_world(d_wide))
    rays.d = -rays.d
    colors_Lo, _, _ = integrator.sample(scene, sampler, rays)
    Lo_per_d = dr.block_reduce(dr.ReduceOp.Add, colors_Lo, spp) / spp

    if mesh.is_emitter():
        # redundant computation, Le is sampled `spp` times but is always the same
        colors_Le = mesh.emitter().eval(si_wide)
        Le_per_d = dr.block_reduce(dr.ReduceOp.Add, colors_Le, spp) / spp

    idx2 = dr.tile(dr.arange(UInt, dirs_per_point), Nv)
    quadW_wide = dr.gather(type(quad_W), quad_W, idx2)
    sh_basis_wide = dr.gather(type(sh_basis), sh_basis, idx2)

    for sh_id, basis_wide in enumerate(sh_basis_wide):
        Lo_coeff = dr.block_reduce(dr.ReduceOp.Add, quadW_wide * basis_wide * Lo_per_d, block_size = dirs_per_point)
        mesh.add_attribute(
            name=f"vertex_Lo_coeffs_{sh_id}",
            size=3,
            buffer=dr.ravel(Lo_coeff))
        
    if mesh.is_emitter():
        for sh_id, basis_wide in enumerate(sh_basis_wide):
            Le_coeff = dr.block_reduce(dr.ReduceOp.Add, quadW_wide * basis_wide * Le_per_d, block_size = dirs_per_point)
            mesh.add_attribute(
                name=f"vertex_Le_coeffs_{sh_id}",
                size=3,
                buffer=dr.ravel(Le_coeff))


def fit_sh_on_mesh_batched(scene: mi.Scene, mesh: mi.Mesh, max_order: int, Nquad: int = 64, spp: int = 32, seed: int = 0):
    Nv = mesh.vertex_count()
    # Initialize interaction structs at each of the mesh vertices
    positions = dr.unravel(mi.Point3f, mesh.vertex_positions_buffer())
    if mesh.has_vertex_normals():
        normals = dr.unravel(mi.Vector3f, mesh.vertex_normals_buffer())
    else:
        V = positions.numpy().T
        F = dr.unravel(mi.Point3u, mesh.faces_buffer()).numpy().T
        normals_ = per_vertex_normals(V, F)
        normals = mi.Vector3f(dr.unravel(dr.auto.Array3f, Float(normals_.ravel())))
    si = dr.zeros(mi.SurfaceInteraction3f, Nv)
    si.p, si.n = positions, normals
    si.sh_frame = mi.Frame3f(normals)

    # evaluate SH basis
    # d, sh_basis, quad_W = eval_basis_on_hemisphere(max_order, Nquad)
    d, sh_basis, quad_W = eval_basis(max_order, Nquad)
    dirs_per_point = dr.width(d)

    # Split computation into batches if needed?
    MAX_RAYS_PER_BATCH = 1 << 30
    rays_per_vtx = dirs_per_point * spp
    total_rays = Nv * dirs_per_point * spp
    # print(f"Nv = {Nv}, Nrays = {total_rays}")
    vtx_batch_size = Nv
    num_batches = (total_rays + MAX_RAYS_PER_BATCH) // MAX_RAYS_PER_BATCH
    # print(f"Num batches: {num_batches}")
    if num_batches > 1:
        vtx_batch_size = MAX_RAYS_PER_BATCH // rays_per_vtx
    
    vtx_idx_start = 0
    vtx_idx_end = vtx_idx_start + vtx_batch_size
    Lo_coeffs = [[] for _ in range(get_sh_count(max_order))]
    Le_coeffs = [[] for _ in range(get_sh_count(max_order))]
    start_idxs, end_idxs = [], []
    for _ in range(num_batches):
        start_idxs.append(vtx_idx_start)
        end_idxs.append(vtx_idx_end)
        # Trace and aggregate rays
        si_wide = dr.gather(type(si), si, dr.repeat(dr.arange(UInt, vtx_idx_start, vtx_idx_end), spp * dirs_per_point))
        d_wide = dr.gather(type(d), d, dr.tile(dr.repeat(dr.arange(UInt, dirs_per_point), spp), vtx_batch_size))
        si_wide.wi = d_wide

        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(seed, vtx_batch_size * dirs_per_point * spp)
        # measure radiance from position `d` towards `si.p`
        # spawn rays. first, spawn the ray towards `d` so that its startpoint is on the surface's exterior
        rays = si_wide.spawn_ray(si_wide.to_world(d_wide))
        # then, reverse `d` so that it points towards `si.p`
        rays.d = -rays.d
        colors_Lo, _, _ = scene.integrator().sample(scene, sampler, rays)
        Lo_per_d = dr.block_reduce(dr.ReduceOp.Add, colors_Lo, spp) / spp

        if mesh.is_emitter():
            # redundant computation, Le is sampled `spp` times but is always the same
            colors_Le = mesh.emitter().eval(si_wide)
            Le_per_d = dr.block_reduce(dr.ReduceOp.Add, colors_Le, spp) / spp

        idx2 = dr.tile(dr.arange(UInt, dirs_per_point), vtx_batch_size)
        quadW_wide = dr.gather(type(quad_W), quad_W, idx2)
        sh_basis_wide = dr.gather(type(sh_basis), sh_basis, idx2)

        for sh_id, basis_wide in enumerate(sh_basis_wide):
            Lo_coeff = dr.block_reduce(dr.ReduceOp.Add, quadW_wide * basis_wide * Lo_per_d, block_size = dirs_per_point)
            # NOTE/TODO: can subsume the block_reduce() and scatter_add() into one scatter_add()
            Lo_coeffs[sh_id].append(Lo_coeff)
            
        if mesh.is_emitter():
            for sh_id, basis_wide in enumerate(sh_basis_wide):
                Le_coeff = dr.block_reduce(dr.ReduceOp.Add, quadW_wide * basis_wide * Le_per_d, block_size = dirs_per_point)
                # NOTE/TODO: can subsume the block_reduce() and scatter_add() into one scatter_add()
                Le_coeffs[sh_id].append(Le_coeff)
                
        vtx_idx_start = vtx_idx_end
        vtx_idx_end = min(Nv, vtx_idx_start + vtx_batch_size)
        vtx_batch_size = vtx_idx_end - vtx_idx_start

    for sh_id, batched_coeffs in enumerate(Lo_coeffs):
        Lo_coeffs = dr.zeros(mi.Color3f, Nv)
        for coeff_for_batch, vtx_start, vtx_end in zip(batched_coeffs, start_idxs, end_idxs):
            # NOTE/TODO: can subsume the block_reduce() and scatter_add() into one scatter_add()
            dr.scatter_add(Lo_coeffs, coeff_for_batch, dr.arange(UInt, vtx_start, vtx_end))

        mesh.add_attribute(
            name=f"vertex_Lo_coeffs_{sh_id}",
            size=3,
            buffer=dr.ravel(Lo_coeffs))

    if not(mesh.is_emitter()):
        return

    for sh_id, batched_coeffs in enumerate(Le_coeffs):
        Le_coeffs = dr.zeros(mi.Color3f, Nv)
        for coeff_for_batch, vtx_start, vtx_end in zip(batched_coeffs, start_idxs, end_idxs):
            # NOTE/TODO: can subsume the block_reduce() and scatter_add() into one scatter_add()
            dr.scatter_add(Le_coeffs, coeff_for_batch, dr.arange(UInt, vtx_start, vtx_end))
        
        mesh.add_attribute(
            name=f"vertex_Le_coeffs_{sh_id}",
            size=3,
            buffer=dr.ravel(Le_coeffs))


def fit_sh_on_scene(scene: mi.Scene, max_order: int):
    meshes = [shape for shape in scene.shapes() if shape.is_mesh()]
    for idx, mesh in enumerate(tqdm(meshes)):
        fit_sh_on_mesh_batched(scene, mesh, max_order, Nquad=32*4, spp=32*2, seed=idx)     # matpreview, 30s


def render_scene(scene: mi.Scene, max_order: int):
    # Render scene
    film_size = scene.sensors()[0].film().size()
    img_res = (film_size[0], film_size[1])
    us, vs = dr.meshgrid(dr.linspace(Float, 0.0, 1.0, img_res[0]), dr.linspace(Float, 0.0, 1.0, img_res[1]))
    sensor = scene.sensors()[0]
    ray, _ = sensor.sample_ray(0.0, 0.0, mi.Point2f(us, vs), dr.zeros(mi.Point2f))
    si = scene.ray_intersect(ray)
    mesh = si.shape

    color = dr.zeros(mi.Color3f, dr.width(us))
    for sh_id, basis in enumerate(dr.sh_eval(si.wi, max_order)):
        # `Lo` already includes the emittance of the surface
        # color += dr.select(mesh.is_emitter(), 
        #     mesh.eval_attribute_3(f"vertex_Le_coeffs_{sh_id}", si) * basis, 
        #     dr.zeros(mi.Color3f))
        Lo_coeff = mesh.eval_attribute_3(f"vertex_Lo_coeffs_{sh_id}", si)
        color += Lo_coeff * basis
    color = dr.clip(color, 0.0, 1.0)

    # out = mi.TensorXf(dr.select(mesh == meshes[4], 0.0, 1.0), shape=img_res)

    # TODO: find out how to plot mi.Color3f images
    out = np.stack((
        mi.TensorXf(color.x, shape=(img_res[1], img_res[0])).numpy().T,
        mi.TensorXf(color.y, shape=(img_res[1], img_res[0])).numpy().T,
        mi.TensorXf(color.z, shape=(img_res[1], img_res[0])).numpy().T)).swapaxes(0,2)

    plt.figure(figsize=(18,10)); 
    plt.subplot(131); plt.imshow(mi.render(scene)); plt.title("Reference")
    prim = mi.render(scene, integrator = mi.load_dict({'type': 'aov', 'aovs': 'dd.y:prim_index'}))
    plt.subplot(132); plt.imshow(prim.numpy().astype('int') % 10); plt.title("Triangles")
    plt.subplot(133); plt.imshow(out); plt.title("Radiance cache")
    plt.tight_layout()

    return color