import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
from sh_fitting import get_sh_count
from radiosity_sh import RadianceCacheMiSH, SceneSurfaceSampler
from radiosity_rt import RadianceCacheMiRT

def balance_heuristic(pdf1: Float, pdf2: Float, n1: int = 1, n2: int = 1):
    return (n1 * pdf1) * dr.rcp(dr.fma(n1, pdf1, n2 * pdf2))

def power_heuristic(pdf1: Float, pdf2: Float, n1: int = 1, n2: int = 1):
    pdf1_ = dr.square(pdf1 * n1)
    pdf2_ = dr.square(pdf2 * n2)
    return pdf1_ * dr.rcp(pdf1_ + pdf2_)

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
            # E_vi has units of (irradiance) ** 2 -- it is the integral of the *squared* radiance Lo over 
            # the hemisphere. 
            E_v1 = dr.gather(dtype, vertex_energies, face_idxs.x)
            E_v2 = dr.gather(dtype, vertex_energies, face_idxs.y)
            E_v3 = dr.gather(dtype, vertex_energies, face_idxs.z)
            # To compute the emitted power of the triangle, take the sqrt() of E_vi and scale by the triangle area.
            E_faces = face_area * dr.rcp(3.0) * (dr.sqrt(E_v1) + dr.sqrt(E_v2) + dr.sqrt(E_v3))

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

    
    def sample(self, si: mi.SurfaceInteraction3f, sample1: Float, sample2: mi.Point2f) -> tuple[mi.Vector3f, Float, Float]:
        # Sample a triangle in the emissive scene
        # NOTE: sample_reuse_pmf() appears to be nondeterministic wrt reused_sample!!!
        global_face_idx, pdf = self.pmf.sample_pmf(sample1)

        # Find the mesh and prim index corresponding to this triangle
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

        # Compute surface data at sample
        pdf_area = pdf / compute_face_areas(mesh, fi)
        sample_pos = dr.fma(e0, b.x, dr.fma(e1, b.y, p0))
        sample_n = dr.select(
            mesh.has_vertex_normals(), 
            dr.fma(mesh.vertex_normal(fi.x), (1.0 - b.x - b.y),
                    dr.fma(mesh.vertex_normal(fi.y), b.x,
                           mesh.vertex_normal(fi.z) * b.y)),
            dr.cross(e0, e1))
        sample_n = dr.normalize(sample_n)

        # Handle geometry term and convert pdf to solid angle measure
        d_world = sample_pos - si.p
        dist_squared = dr.squared_norm(d_world)
        d_world *= dr.rsqrt(dist_squared)
        cos_theta_o = dr.abs(dr.dot(sample_n, -d_world))
        G = dr.select(dist_squared > 0.0, cos_theta_o * dr.rcp(dist_squared), 0.0)
        pdf_angle = dr.select(G > 0.0, pdf_area * dr.rcp(G), 0.0)

        # Handle visibility
        vis_ray = si.spawn_ray_to(sample_pos)
        occluded = self.scene.ray_test(vis_ray)

        # MC weight: vis / l_pdf_angle
        weight = dr.select(~occluded & (pdf_angle > 0.0), dr.rcp(pdf_angle), 0.0)
        d_local = si.to_local(d_world)
        return d_local, weight, pdf_angle
    
    def eval_pdf(self, si: mi.SurfaceInteraction3f, wi_local: mi.Vector3f) -> Float:
        # Launch rays along `wi` and identify the hit emissive triangle
        d_world = si.to_world(wi_local)
        ray = si.spawn_ray(d_world)
        pi = self.scene.ray_intersect_preliminary(ray)
        active = pi.is_valid()
        local_face_idx = pi.prim_index

        # Find the index of the hit triangle in the energy PMF
        mesh_idx = dr.zeros(dr.cuda.ad.UInt, dr.width(wi_local))
        mesh_ptrs = self.scene.shapes_dr()
        for i, shape in enumerate(mesh_ptrs):
            mesh_idx[pi.shape == shape] = i
        idx_offsets = dr.gather(type(self.mesh_start_idxs), self.mesh_start_idxs, mesh_idx)
        global_face_idx = dr.select(active, local_face_idx + idx_offsets, 0)
        
        # Evaluate the hit probability
        meshes = dr.gather(mi.MeshPtr, mesh_ptrs, mesh_idx)
        fi = meshes.face_indices(local_face_idx)
        pdf = dr.select(active, 
                        self.pmf.eval_pmf_normalized(global_face_idx) / compute_face_areas(meshes, fi),
                        Float(0.0))
        
        # Compute geometry term for this hit point
        dist = pi.t
        si_triangle = pi.compute_surface_interaction(ray, active=active)
        # TODO: sh_frame.N or si.N?
        # cos_theta_o = dr.abs(dr.dot(si_triangle.sh_frame.n, -d_world))
        cos_theta_o = dr.abs(dr.dot(si_triangle.n, -d_world))   
        G = dr.select(active & (dist > 0.0), cos_theta_o * dr.rcp(dist * dist), 0.0)

        # Convert area pdf to solid angle measure
        pdf_angle = dr.select(G > 0.0, pdf * dr.rcp(G), 0.0)
        return pdf_angle
    
    def test(self, si: mi.SurfaceInteraction3f, sample1: Float, sample2: mi.Point2f):
        wi_local, _, pdf_ref, G_ref = self.sample(si, sample1, sample2)
        pdf, G = self.eval_pdf(si, wi_local)
        print(pdf_ref)
        print(pdf)
        print(G_ref)
        print(G)
        print("pdf close: ", dr.allclose(pdf, pdf_ref))
        print("G close: ", dr.allclose(G, G_ref))
        print((pdf - pdf_ref).numpy())


        
import numpy as np

class RadianceCacheEM(RadianceCacheMiRT):
    def __init__(self, scene: mi.Scene, spp_per_wo: int, spp_per_wi: int):
        super().__init__(scene, spp_per_wo, spp_per_wi)
        self.energy_pmf = EnergyPMF(RadianceCacheMiSH(scene))

    def pathtrace(self, rays: mi.Ray3f, spp: int, sampler_rt: mi.Sampler, rng_state: int, active: Bool = None) -> tuple[mi.Color3f, int]:
        '''
        Inputs:
            - TODO
        Outputs:
            - L: mi.Color3f. Incident radiances along `rays`, computed via pathtracing.
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        if active is None:
            active = dr.full(Bool, True, dr.width(rays))
        num_rays = dr.width(rays)
        ray_flat_idxs = dr.repeat(dr.arange(UInt, num_rays), spp)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, rays, ray_flat_idxs, mode = dr.ReduceMode.Local)
        active_flattened = dr.gather(type(active), active, ray_flat_idxs, mode = dr.ReduceMode.Local)
        sampler_rt.seed(rng_state, num_rays * spp); rng_state += 0x00FF_FFFF
        colors, _, _ = self.integrator.sample(self.scene, sampler_rt, rays_flattened, active = active_flattened)
        L = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = spp) / spp
        return L, rng_state

    def query_cached_Lo(self, si: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0)  -> tuple[mi.Color3f, Bool, int]:
        '''
        Inputs:
            - sampler: Sampler. The pseudo-random number generator.
            - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
        Outputs: 
            - Lo: mi.Color3f. Array of outgoing radiances of size [#si,].
            - active: dr.Bool. Active lanes.
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        # Compute the outgoing radiance from `A` for a direction, `wo`
        # Note that `wo` is stored in the `si.wi` field (unintuitive, but needed for BSDF.eval() later)
        wo_local = si.wi
        wo_world = si.to_world(wo_local)
        Lo_rays = si.spawn_ray(wo_world)
        Lo_rays.d = -Lo_rays.d
        # The ray that's spawned from `si` should intersect `si.shape` again. Due to RayEpsilons and 
        # spawn offsets, there exist edge cases where this does not occur: for example, when `si` lies
        # on the boundary edge of a rectangular plane. If this occurs, we should omit this `si` from 
        # the loss calculation.
        active = (self.scene.ray_intersect_preliminary(Lo_rays).shape == si.shape)

        # Pathtrace along `-wo` to get the radiance when looking at `A`. For each `Lo_ray`,
        # compute `SPP_LO` different pathtraced samples and average them to get the outgoing 
        # radiance.
        Lo, rng_state = self.pathtrace(Lo_rays, self.spp_per_wo, sampler_rt, rng_state, active)
        return Lo, active, rng_state


    def query_cached_Li(self, si_wide: mi.SurfaceInteraction3f, sampler_rt: mi.Sampler, rng_state: int = 0, emitter_sampling: bool = True) \
        -> tuple[mi.Color3f, mi.Vector3f, Bool, int]:
        '''
        Inputs:
            - si_wide: SurfaceInteraction3f. Widened array of surface sample points of size [#si * #wi,].
            - sampler: Sampler. The pseudo-random number generator.
            - rng_state: int. The RNG seed.
        Outputs: 
            - Li: mi.Color3f. Flattened array of incident radiances of size [#si * #wi,]. The data 
            is in contiguous order, i.e. the first #wi entries belong to si0, and so on.
            - wi_local: mi.Vector3f. Flattened array of incident directions of size [#si * #wi,].
            - active: dr.Bool. Active lanes.
            - rng_state: int. RNG seed for the next operation involving random numbers.
        '''
        sampler_rt.seed(rng_state, dr.width(si_wide)); rng_state += 0x00FF_FFFF
        uv = sampler_rt.next_2d()

        if emitter_sampling:
            # light_pdf is expressed in units of solid angle
            em_wi, em_weight, em_pdf = self.energy_pmf.sample(si_wide, sampler_rt.next_1d(), uv)
            # assert ~dr.any((light_pdf == 0.0) & (light_weight != 0.0))

            # Evaluate material pdf and MIS weight
            hemi_pdf = mi.warp.square_to_cosine_hemisphere_pdf(em_wi)
            mis_weight = dr.select(hemi_pdf > 0.0, balance_heuristic(em_pdf, hemi_pdf), 1.0)
            em_weight *= mis_weight 
            wi_local, wi_pdf, wi_weight = em_wi, em_pdf, em_weight

            # # DEBUG
            # self.energy_pmf.test(si_wide, sampler_rt.next_1d(), sampler_rt.next_2d())
        else:
            hemi_wi = mi.warp.square_to_cosine_hemisphere(uv)
            hemi_pdf = mi.warp.square_to_cosine_hemisphere_pdf(hemi_wi)
            hemi_weight = dr.select(hemi_pdf > 0.0, dr.rcp(hemi_pdf), 0.0)
            # assert ~dr.any((hemi_pdf == 0.0) & (hemi_weight != 0.0))

            # Evaluate light pdf and MIS weight
            # light_pdf is expressed in units of solid angle
            em_pdf = self.energy_pmf.eval_pdf(si_wide, hemi_wi)
            mis_weight = dr.select(em_pdf > 0.0, balance_heuristic(hemi_pdf, em_pdf), 1.0)
            hemi_weight *= mis_weight
            wi_local, wi_pdf, wi_weight = hemi_wi, hemi_pdf, hemi_weight

        assert ~dr.any((wi_pdf == 0.0) & (wi_weight != 0.0))

        wi_rays = si_wide.spawn_ray(si_wide.to_world(wi_local))
        active = wi_weight > 0.0

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        Li, rng_state = self.pathtrace(wi_rays, self.spp_per_wi, sampler_rt, rng_state, active)
        Li *= wi_weight
        return Li, wi_local, active, rng_state #, wi_rays
    

from visualizer import plot_rays
from vertex_bsdf import visualize_textures
import polyscope as ps

def compute_loss_DEBUG(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCacheEM, 
        trainable_bsdf: mi.BSDF, 
        num_points: int = 1,
        num_wi: int = 256, 
        rng_state: int = 0,
        plot: bool = False
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
        si, delta_emitter_sample, delta_emitter_Li, rng_state = scene_sampler.sample(num_points, sampler_rt, rng_state)

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
        Le = radiance_cache.query_cached_Le(si)
        Lo, active_si, rng_state = radiance_cache.query_cached_Lo(si, sampler_rt, rng_state)
        lhs = -Le + Lo

        # Build the "wide" `si`
        #     For each surface point `si`, we should sample `num_wi` incident directions.
        # `wi` can be thought of as a 2D matrix[NUM_POINTS, num_wi] while `si` is an 
        # array[NUM_POINTS]. The latter needs to be broadcasted to match the shape of `wi`, 
        # which is done using the `gather()` (aka "widen") operation.
        #
        #     `si_wide` has the form:          v---- NUM_WI copies ---v
        # [s0, ..., s0, s1, ..., s1,    ...   sN-1,      ...,       sN-1]   (contiguous order)
        num_points = dr.width(si)
        si_wide = dr.gather(type(si), si, 
                            index = dr.repeat(dr.arange(UInt, num_points), num_wi), 
                            mode = dr.ReduceMode.Local)

        # Evaluate RHS integral
        # Light sampling
        Li_em, wi_em, active_em, rng_state = radiance_cache.query_cached_Li(si_wide, sampler_rt, rng_state, True)

        # Material/cosine sampling
        Li_mat, wi_mat, active_mat, rng_state = radiance_cache.query_cached_Li(si_wide, sampler_rt, rng_state, False)

        with dr.resume_grad():
            integrand = Li_mat * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_mat, active = active_mat) \
                       + Li_em * trainable_bsdf.eval(ctx, si = si_wide, wo = wi_em,  active = active_em)
            rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi
            residuals = dr.select(active_si, dr.squared_norm(lhs - rhs), 0.0)
            loss = 0.5 * dr.mean(residuals)



            if False: #plot or (loss.numpy().item() > 0.01):
                err = dr.squared_norm(lhs - rhs).numpy()
                idx = np.where(err > 0.01)[0]
                bad_si = dr.gather(mi.SurfaceInteraction3f, si, idx)
                
                # print(np.histogram(err, bins = np.logspace(-4,2, base=10, num=13)))
                print(f"Max error at index {idx} (err = {err[idx]}).")
                print(f"lhs = {lhs.numpy()[:,idx]}")
                print(f"rhs = {rhs.numpy()[:,idx]}")
                print(bad_si)
                print(f"RNG: {rng_state}")

                ps.init()
                visualize_textures(radiance_cache.scene, False)
                plot_rays(wi_rays, "wi")
                si_cloud = ps.register_point_cloud("si", si.p.numpy().T)
                si_cloud.add_vector_quantity("wo", si.to_world(si.wi).numpy().T)
                points = ps.register_point_cloud("Bad si", bad_si.p.numpy().T)
                points.add_vector_quantity("wo", bad_si.to_world(bad_si.wi).numpy().T)
                ps.show()
            
            return loss