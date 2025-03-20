import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
from sh_fitting import get_sh_count
from radiosity_sh import RadianceCacheMiSH, SceneSurfaceSampler
from radiosity_rt import RadianceCacheMiRT

def compute_face_areas(mesh: mi.Mesh, face_idxs: mi.Point3u):
        p0 = mesh.vertex_position(face_idxs.x)
        p1 = mesh.vertex_position(face_idxs.y)
        p2 = mesh.vertex_position(face_idxs.z)
        e0, e1 = p1 - p0, p2 - p0
        return 0.5 * dr.norm(dr.cross(e0, e1))

class EnergyPMF:
    def __init__(self, rc: RadianceCacheMiSH):
        self.build_energy_pmf(rc)
        self.meshes = rc.scene.shapes_dr()
        self.scene = rc.scene

    def build_energy_pmf(self, rc: RadianceCacheMiSH) -> mi.DiscreteDistribution:
        meshes = rc.scene.shapes_dr()
        mesh_active = meshes.is_mesh()
        face_counts = dr.cuda.ad.UInt(dr.select(mesh_active, mi.MeshPtr(meshes).face_count(), 0))
        mesh_start_indices = dr.prefix_reduce(dr.ReduceOp.Add, face_counts, exclusive=True)
        mesh_end_indices   = dr.prefix_reduce(dr.ReduceOp.Add, face_counts, exclusive=False)

        E_scene = dr.zeros(Float, mesh_end_indices[-1])
        for mesh_idx, mesh in enumerate(meshes):
            # Compute the outgoing energy from each mesh vertex as the sum of the squared SH coefficients
            vertex_energies = dr.zeros(Float, mesh.vertex_count())
            for sh_idx in range(get_sh_count(rc.order)):
                vertex_coeffs = dr.unravel(mi.Color3f, mesh.attribute_buffer(f"vertex_Lo_coeffs_{sh_idx}"))
                vertex_energies += dr.squared_norm(vertex_coeffs)

            E_faces = dr.zeros(Float, mesh.face_count())
            face_idxs = dr.unravel(mi.Point3u, mesh.faces_buffer())
            face_area = compute_face_areas(mesh, face_idxs)
            dtype = type(vertex_energies)
            E_v1 = dr.gather(dtype, vertex_energies, face_idxs.x)
            E_v2 = dr.gather(dtype, vertex_energies, face_idxs.y)
            E_v3 = dr.gather(dtype, vertex_energies, face_idxs.z)
            E_faces = face_area * dr.rcp(3.0) * (E_v1 + E_v2 + E_v3)

            # Store the mesh faces' energy values in the scene-level energy-per-vertex array
            start_idx = mesh_start_indices[mesh_idx]
            end_idx = mesh_end_indices[mesh_idx]
            dr.scatter_add(E_scene, E_faces, dr.arange(UInt, start_idx, end_idx))

        # # DEBUG: the scene-level array `E_scene` should simply be the concatenation of the meshes' energy-per-vertex arrays
        # import numpy as np
        # E_scene_np = []
        # for mesh_idx, mesh in enumerate(meshes):
        #     E_mesh = dr.zeros(Float, mesh.vertex_count())
        #     for sh_idx in range(get_sh_count(rc.order)):
        #         coeffs = dr.unravel(mi.Color3f, mesh.attribute_buffer(f"vertex_Lo_coeffs_{sh_idx}"))
        #         E_mesh += dr.squared_norm(coeffs)
        #     E_scene_np.append(E_mesh.numpy())

        # E_scene_np = np.concatenate(E_scene_np)
        # assert np.allclose(E_scene_np, E_scene.numpy()), "Scene vertex array is not correct!"

        pmf = mi.DiscreteDistribution(E_scene)
        
        self.pmf = pmf
        self.mesh_start_idxs = mesh_start_indices 
        self.mesh_end_idxs = mesh_end_indices

    
    def sample(self, si: mi.SurfaceInteraction3f, sample2_: mi.Point2f) -> tuple[mi.Vector3f, Float, Float]:
        # Sample a triangle on the emissive scene
        sample2 = mi.Point2f(sample2_)
        global_face_idx, sample_x, pdf = self.pmf.sample_reuse_pmf(sample2.x)
        sample2.x = sample_x

        # find the mesh and prim index corresponding to this triangle
        dtype = type(self.mesh_end_idxs)
        mesh_count = len(self.mesh_end_idxs)
        mesh_idxs = dr.binary_search(0, mesh_count, lambda index: dr.gather(dtype, self.mesh_end_idxs, index) <= global_face_idx)
        idx_offset = dr.gather(dtype, self.mesh_start_idxs, mesh_idxs)
        local_face_idx = global_face_idx - idx_offset

        # Sample a point on the triangle
        mesh = dr.gather(mi.MeshPtr, self.meshes, mesh_idxs)
        fi = mesh.face_indices(local_face_idx)

        p0 = mesh.vertex_position(fi.x)
        p1 = mesh.vertex_position(fi.y)
        p2 = mesh.vertex_position(fi.z)
        e0 = p1 - p0
        e1 = p2 - p0
        b = mi.warp.square_to_uniform_triangle(sample2)

        ps = dr.zeros(mi.PositionSample3f, dr.width(sample2))
        ps.time  = 0.0
        ps.pdf   = pdf
        ps.delta = False

        ps.p     = dr.fma(e0, b.x, dr.fma(e1, b.y, p0))
        ps.n = dr.select(
            mesh.has_vertex_normals(), 
            dr.fma(mesh.vertex_normal(fi.x), (1.0 - b.x - b.y),
                    dr.fma(mesh.vertex_normal(fi.y), b.x,
                           mesh.vertex_normal(fi.z) * b.y)),
            dr.cross(e0, e1))
        ps.n = dr.normalize(ps.n)

        d_world = ps.p - si.p
        dist_squared = dr.squared_norm(d_world)
        d_world *= dr.rsqrt(dist_squared)

        # standard area sampling case
        # weight = dr.rcp(self.pmf.normalization())     # sum of surface areas

        # energy-weighted area sampling?
        # weight = self.pmf.sum() / (self.pmf.eval_pmf(global_face_idx) / compute_face_areas(mesh, fi))
        # weight = dr.select(pdf > 0.0, dr.rcp(pdf) * compute_face_areas(mesh, fi), 0.0)
        ps.pdf /= compute_face_areas(mesh, fi)
        weight = dr.select(ps.pdf > 0.0, dr.rcp(ps.pdf), 0.0)

        # Handle geometry term?
        weight *= dr.select(dist_squared > 0.0, dr.rcp(dist_squared), 0.0)
        cos_theta_o = dr.maximum(0.0, dr.dot(ps.n, -d_world))
        weight *= cos_theta_o               # ?
        # cos_theta_i = dr.maximum(0.0, mi.Frame3f.cos_theta(si.wi))
        # weight *= cos_theta_i               # ?

        vis_ray = si.spawn_ray(d_world)
        vis_ray.maxt = dr.sqrt(dist_squared) - 2.0 * mi.math.RayEpsilon
        weight &= ~self.scene.ray_test(vis_ray)

        d_local = si.to_local(d_world)

        return d_local, weight, pdf

class RadianceCacheEM(RadianceCacheMiRT):
    def __init__(self, scene: mi.Scene, spp_per_wo: int, spp_per_wi: int):
        super().__init__(scene, spp_per_wo, spp_per_wi)
        self.energy_pmf = EnergyPMF(RadianceCacheMiSH(scene))

    def pathtrace(self, rays: mi.Ray3f, spp: int, sampler_rt: mi.Sampler, rng_state: int):
        num_rays = dr.width(rays)
        ray_flat_idxs = dr.repeat(dr.arange(UInt, num_rays), spp)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, rays, ray_flat_idxs, mode = dr.ReduceMode.Local)
        sampler_rt.seed(rng_state, num_rays * spp)
        colors, _, _ = self.integrator.sample(self.scene, sampler_rt, rays_flattened)
        L = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = spp) / spp
        return L

    def query_cached_Lo(self, si: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0)  -> mi.Color3f:
        '''
        Inputs:
            - sampler: Sampler. The pseudo-random number generator.
            - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
        Outputs: 
            - Lo: mi.Color3f. Array of outgoing radiances of size [#si,].
        '''
        # Compute the outgoing radiance from `A` for a direction, `wo`
        # Note that `wo` is stored in the `si.wi` field (unintuitive, but needed for BSDF.eval() later)
        wo_local = si.wi
        wo_world = si.to_world(wo_local)
        Lo_rays = si.spawn_ray(wo_world)
        Lo_rays.d = -Lo_rays.d

        # Pathtrace along `-wo` to get the radiance when looking at `A`. For each `Lo_ray`,
        # compute `SPP_LO` different pathtraced samples and average them to get the outgoing 
        # radiance.
        return self.pathtrace(Lo_rays, self.spp_per_wo, sampler_rt, rng_state + 0x0FFF_FFFF)


    def query_cached_Li(self, si: mi.SurfaceInteraction3f, num_wi: int, sampler_rt: mi.Sampler, rng_state: int = 0) -> mi.Color3f:
        '''
        Inputs:
            - sampler: Sampler. The pseudo-random number generator.
            - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
            - num_wi: int. Number of incident directions `wi` to use at each surface point.
            - spp: int. Number of samples to use per evaluation of Li.
        Outputs: 
            - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
            is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
            - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
            - si_flattened: mi.Vector3f. Flattened array of surface sample points of size [#si * #wi,].
        '''
        # For each surface point, we should sample `NUM_DIRS` different `wi` directions.
        # For now, assume that the points do NOT use identical `wi` directions, i.e. we
        # need to draw a total of `NUM_DIRS * NUM_POINTS` samples for `wi`. 
        # In that case, `wi` is a 2D matrix[NUM_POINTS, NUM_DIRS] while `si` is an 
        # array[NUM_POINTS]. The latter needs to be broadcasted to match the shape of 
        # `wi`, which is done using the `gather()` (aka "flatten") operation.
        #
        # There is one possible simplification with unknown impact on correctness: use 
        # the same set of local `wi` directions at every point.
        # Code changes: sample only `NUM_DIR` vectors for `wo_local`. Subsequently,
        # we still need to take the "outer product" of the `wo_local` array with the 
        # `si` array to get `NUM_POINTS * NUM_DIRS` rays.
        #
        num_points = dr.width(si)
        # the `flat_idxs` has the form: 
        #                                      v---- NUM_WI copies ---v
        # [0, ..., 0, 1, ..., 1,    ...    NUM_POINTS-1, ..., NUM_POINTS-1]   (contiguous order)
        #
        si_flat_idxs = dr.repeat(dr.arange(UInt, num_points), num_wi)
        # `si_flattened` has the form:
        # [s0, ..., s0, s1, ..., s1,    ...    sN-1, ..., sN-1]
        si_flattened = dr.gather(mi.SurfaceInteraction3f, si, si_flat_idxs, mode = dr.ReduceMode.Local)

        # Compute the incident radiance on `A` for a direction, `wi`.
        # Total rays: num_points * num_wi (per point)
        sampler_rt.seed(rng_state + 2 * 0x0FFF_FFFF, num_points * num_wi)
        uv = sampler_rt.next_2d()

        light_wi, light_weight, light_pdf = self.energy_pmf.sample(si_flattened, uv)
        # light_eval_pdf = mi.warp.square_to_cosine_hemisphere_pdf(light_wi)


        
        # wi_local = mi.warp.square_to_cosine_hemisphere(uv)
        # wi_pdf   = mi.warp.square_to_cosine_hemisphere_pdf(wi_local)
        # wi_weight = dr.select(wi_pdf > 0.0, dr.rcp(wi_pdf), 0.0)

        wi_rays = si_flattened.spawn_ray(si_flattened.to_world(light_wi))

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        Li = self.pathtrace(wi_rays, self.spp_per_wi, sampler_rt, rng_state + 3 * 0x0FFF_FFFF)
        Li *= light_weight

        return Li, light_wi, si_flattened, wi_rays
    

from visualizer import plot_rays
import polyscope as ps

def compute_loss_DEBUG(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCacheMiRT, 
        trainable_bsdf: mi.BSDF, 
        num_points: int = 1,
        num_wi: int = 256, 
        rng_state: int = 0,
        ):
    '''
    Inputs:
        - scene_sampler: SceneSurfaceSampler. The scene sampler draws random points from the scene's surfaces.
        - radiance_cache: RadianceCache. Data structure containing the emissive surface data.
        - trainable_bsdf: mi.BSDF. 
        - num_points: int. The number of surface point samples to use.
        - num_wi: int. The number of incident directions per surface point to use to calculate the radiosity integral.
    Outputs:
        - loss: Float. The scalar loss.
    
    # NOTE: we might be able to further re-arrange the steps to cut down/consolidate kernel launches.
    '''
    with dr.suspend_grad():
        # Temp workaround. TODO: avoid initializing a new sampler at each iteration
        sampler_rt: mi.Sampler = mi.load_dict({'type': 'independent'})

        # Sample `NUM_POINTS` different surface points
        si, delta_emitter_sample, delta_emitter_Li = scene_sampler.sample(num_points, sampler_rt, rng_state)
        # print(si)
        # print(si.shape)

        # Evaluate RHS scene emitter contribution
        ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)

        # perform a ray visibility test from `si` to the delta emitter
        vis_rays = si.spawn_ray(delta_emitter_sample.d)
        vis_rays.maxt = delta_emitter_sample.dist
        emitter_occluded = radiance_cache.scene.ray_test(vis_rays)
        delta_emitter_Li &= ~emitter_occluded
        with dr.resume_grad():
            f_emitter = trainable_bsdf.eval(ctx, si, wo = si.to_local(delta_emitter_sample.d))
            rhs = f_emitter * delta_emitter_Li

        # Evaluate LHS of balance equation
        lhs = -radiance_cache.query_cached_Le(si)

        lhs += radiance_cache.query_cached_Lo(si, sampler_rt, rng_state)

        # Evaluate RHS integral
        Li, wi_local, si_flattened, wi_rays = radiance_cache.query_cached_Li(si, num_wi, sampler_rt, rng_state)
        # print(f"{dr.width(Li)=}")
        # print(Li)

        # plot_rays(wi_rays, "wi")

        with dr.resume_grad():
            f_io = trainable_bsdf.eval(ctx, si = si_flattened, wo = wi_local)
            integrand = f_io * Li
            rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
            return 0.5 * dr.mean(dr.squared_norm(lhs - rhs))