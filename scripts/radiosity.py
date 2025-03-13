import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt

# High level organization. The loss computation is split into three kernels:
# - Sample geometry points and Lo ray directions
# - Compute Lo
# - Compute Li

def is_delta_emitter(emitter: mi.Emitter):
    return (emitter.m_flags & mi.EmitterFlags.Delta) | \
           (emitter.m_flags & mi.EmitterFlags.DeltaPosition) | \
           (emitter.m_flags & mi.EmitterFlags.DeltaDirection)

class SceneSurfaceSampler:
    def __init__(self, scene: mi.Scene):
        shape_ptrs = scene.shapes_dr()
        # TODO: can't figure out how to concatenate the areas (== dr.Float of size [1,])
        areas = [shape.surface_area().numpy().item() for shape in shape_ptrs]
        areas_dr = Float(areas)
        self.distribution = mi.DiscreteDistribution(areas_dr)
        self.shape_ptrs = shape_ptrs
        self.delta_emitters = [emitter for emitter in scene.emitters() if is_delta_emitter(emitter)]

    def sample(self, num_points: int, sampler_rt: mi.Sampler, rng_state: int = 0) -> mi.SurfaceInteraction3f:
        '''
        Inputs:
            - num_points: int. Number of surface points to sample.
            - sampler: Sampler. The pseudo-random number generator.
            - rng_state: int. Seed for the PRNG.
        Outputs: 
            - si: SurfaceInteraction3f. Array of surface sample points of size [#si,].
            - em_ds: DirectionSample3f. Direction sample records from the delta emitter -> surface samples, size [#si,].
            - em_Li: mi.Color3f. Incident radiances from the delta emitter to the surface samples, size [#si,].
        '''
        # Generate `NUM_POINTS` different surface samples
        sampler_rt.seed(rng_state, num_points)
        idx = self.distribution.sample(sampler_rt.next_1d(), True)
        shape = dr.gather(mi.ShapePtr, self.shape_ptrs, idx)
        uv = sampler_rt.next_2d()
        si = shape.eval_parameterization(uv, active=True)
        # # NOTE: the sampling method below might be more general
        # pos_sample = shape.sample_position(time=0.0, sample=sampler.next_2d())
        # si = mi.SurfaceInteraction3f(pos_sample, dr.zeros(mi.Color0f))

        # Generate one outgoing ray per surface sample
        uv = sampler_rt.next_2d()
        wo_local = mi.warp.square_to_cosine_hemisphere(uv)
        si.wi = wo_local

        # Compute the Li contribution from delta emitter sources
        # This can eventually be modified/extended to handle envmaps
        if len(self.delta_emitters) > 0:
            point_light = self.delta_emitters[0]
            # NOTE: sample `uv` is not actually used in the case of point lights
            uv = sampler_rt.next_2d()
            return si, *point_light.sample_direction(si, uv, True)
        else:
            return si, dr.zeros(mi.DirectionSample3f), dr.zeros(mi.Color3f)
    
class RadianceCacheMITSUBA:
    def __init__(self, scene: mi.Scene, spp_per_wo: int, spp_per_wi: int):
        '''
        Inputs:
            - scene: Scene. The Mitsuba scene.
            - spp_per_wo: int. The number of pathtrace samples to use per Lo ray.
            - spp_per_wi: int. The number of pathtrace samples to use per Li ray.
        '''
        self.scene = scene
        self.integrator = scene.integrator()
        self.spp_per_wo = spp_per_wo
        self.spp_per_wi = spp_per_wi

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
        num_points = dr.width(si)
        wo_local = si.wi
        wo_world = si.to_world(wo_local)
        Lo_rays = si.spawn_ray(wo_world)
        Lo_rays.d = -Lo_rays.d

        # Pathtrace along `-wo` to get the radiance when looking at `A`. For each `Lo_ray`,
        # compute `SPP_LO` different pathtraced samples and average them to get the outgoing 
        # radiance.
        ray_flat_idxs = dr.repeat(dr.arange(UInt, num_points), self.spp_per_wo)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, Lo_rays, ray_flat_idxs, mode = dr.ReduceMode.Direct)
        sampler_rt.seed(rng_state + 0x0FFF_FFFF, num_points * self.spp_per_wo)
        colors, _, _ = self.integrator.sample(self.scene, sampler_rt, rays_flattened)
        Lo = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = self.spp_per_wo) / self.spp_per_wo
        return Lo

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
        #
        # NOTE: `dr.ReduceMode` is not used for the `gather()` itself, but instead to 
        # implement its adjoint operation (scatter) for reverse-mode AD. For our context,
        # the choice doesn't matter since we never backprop through this part of the algorithm,
        # but `ReduceMode.Local` is probably optimal when our arrays are laid out 
        # contiguously, as we do here.
        si_flattened = dr.gather(mi.SurfaceInteraction3f, si, si_flat_idxs, mode = dr.ReduceMode.Local)

        # Compute the incident radiance on `A` for a direction, `wi`
        NUM_RAYS = num_points * num_wi
        sampler_rt.seed(rng_state + 2 * 0x0FFF_FFFF, NUM_RAYS)
        uv = sampler_rt.next_2d()
        wi_local = mi.warp.square_to_cosine_hemisphere(uv)
        wi_pdf   = mi.warp.square_to_cosine_hemisphere_pdf(wi_local)
        wi_world = si_flattened.to_world(wi_local)
        wi_rays = si_flattened.spawn_ray(wi_world)

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        ray_flat_idxs = dr.repeat(dr.arange(UInt, NUM_RAYS), self.spp_per_wi)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, wi_rays, ray_flat_idxs, mode = dr.ReduceMode.Local)
        sampler_rt.seed(rng_state + 3 * 0x0FFF_FFFF, NUM_RAYS * self.spp_per_wi)
        colors, _, _ = self.integrator.sample(self.scene, sampler_rt, rays_flattened)
        Li = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = self.spp_per_wi) / self.spp_per_wi
        Li = dr.select(wi_pdf > 0.0, Li * dr.rcp(wi_pdf), dr.zeros(mi.Color3f))
        return Li, wi_local, si_flattened
    
    def query_cached_Le(self, si: mi.SurfaceInteraction3f) -> mi.Color3f:
        mesh = si.shape
        return dr.select(mesh.is_emitter(), mesh.emitter().eval(si), dr.zeros(mi.Color3f))
    

def compute_loss(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCacheMITSUBA, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        num_wi: int, 
        rng_state: int
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

        # Evaluate RHS scene emitter contribution
        ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)
        with dr.resume_grad():
            f_emitter = trainable_bsdf.eval(ctx, si, wo = si.to_local(delta_emitter_sample.d))
            rhs = f_emitter * delta_emitter_Li

        # Evaluate LHS of balance equation
        lhs = -radiance_cache.query_cached_Le(si)

        lhs += radiance_cache.query_cached_Lo(si, sampler_rt, rng_state)

        # Evaluate RHS integral
        Li, wi_local, si_flattened = radiance_cache.query_cached_Li(si, num_wi, sampler_rt, rng_state)

        with dr.resume_grad():
            f_io = trainable_bsdf.eval(ctx, si = si_flattened, wo = wi_local)
            integrand = f_io * Li
            rhs += dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi

            # # # DEBUG
            # print(f"Li:{rhs}")
            # # print(f"Light contribution: {point_Li}")
            # # print(f"Light contribution an: {albedo * dr.inv_pi * I / (r * r)}")
            # print(f"Lo:{lhs}")
            # # albedo = mi.Color3f([0.2, 0.25, 0.7])
            # # # L_an = albedo * dr.rcp(1.0 - albedo) * intensity / (dr.pi * r ** 2)
            # # L_an = delta_emitter_Li * dr.rcp(1.0 - albedo)
            # # print(f"Lo_an:{L_an}")
            # # print(L_an/lhs)

            return 0.5 * dr.mean(dr.squared_norm(lhs - rhs))