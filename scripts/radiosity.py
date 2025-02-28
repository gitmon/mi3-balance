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

    def sample(self, sampler: mi.Sampler, num_points: int) -> mi.SurfaceInteraction3f:
        '''
        TODO
        '''
        # Generate `NUM_POINTS` different surface samples
        # TODO: the seed value matters! Can we change the Sampler width without resetting the seed?
        sampler.seed(0, num_points)
        idx = self.distribution.sample(sampler.next_1d(), True)
        shape = dr.gather(mi.ShapePtr, self.shape_ptrs, idx)
        si = shape.eval_parameterization(sampler.next_2d(), active=True)
        # TODO: the method below should give more uniform sampling, if we can get it to work
        # pos_sample = shape.sample_position(time=0.0, sample=sampler.next_2d())
        # si = mi.SurfaceInteraction3f(pos_sample, dr.zeros(mi.Color0f))

        # Generate one outgoing ray per surface sample
        wo_local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
        si.wi = wo_local

        # Compute the Li contribution from delta emitter sources
        # This can eventually be modified/extended to handle envmaps
        point = self.delta_emitters[0]
        return si, point.sample_direction(si, sampler.next_2d(), True)
    
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

    def query_cached_Lo(self, sampler: mi.Sampler, si: mi.SurfaceInteraction3f)  -> mi.Color3f:
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
        sampler.seed(1, num_points * self.spp_per_wo)
        colors, _, _ = self.integrator.sample(self.scene, sampler, rays_flattened)
        Lo = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = self.spp_per_wo) / self.spp_per_wo
        return Lo

    def query_cached_Li(self, sampler: mi.Sampler, si: mi.SurfaceInteraction3f, num_wi: int) -> mi.Color3f:
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
        sampler.seed(2, num_points * num_wi)
        # the `flat_idxs` has the form: 
        #                                      v---- NUM_WI copies ---v
        # [0, ..., 0, 1, ..., 1,    ...    NUM_POINTS-1, ..., NUM_POINTS-1]   (contiguous order)
        #
        si_flat_idxs = dr.repeat(dr.arange(UInt, num_points), num_wi)
        # `si_flattened` has the form:
        # [s0, ..., s0, s1, ..., s1,    ...    sN-1, ..., sN-1]
        #
        # NOTE: `dr.ReduceMode` is not used for the `gather()` itself, but instead to 
        # implement its adjoint operation (scatter) for reverse-mode AD. For this context,
        # the choice doesn't matter since we never backprop through this part of the algorithm,
        # but `ReduceMode.Local` is probably optimal when our arrays are laid out 
        # contiguously, as we do here.
        si_flattened = dr.gather(mi.SurfaceInteraction3f, si, si_flat_idxs, mode = dr.ReduceMode.Local)

        # Compute the incident radiance on `A` for a direction, `wi`
        wi_local = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
        wi_world = si_flattened.to_world(wi_local)
        wi_rays = si_flattened.spawn_ray(wi_world)

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        NUM_RAYS = num_points * num_wi
        ray_flat_idxs = dr.repeat(dr.arange(UInt, NUM_RAYS), self.spp_per_wi)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, wi_rays, ray_flat_idxs, mode = dr.ReduceMode.Local)
        sampler.seed(3, NUM_RAYS * self.spp_per_wi)
        colors, _, _ = self.integrator.sample(self.scene, sampler, rays_flattened)
        Li = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = self.spp_per_wi) / self.spp_per_wi
        return Li, wi_local, si_flattened

def compute_loss(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCacheMITSUBA, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        num_wi: int):
    '''
    Inputs:
        - scene_sampler: SceneSurfaceSampler. The scene sampler draws random points from the scene's surfaces.
        - radiance_cache: RadianceCache. Data structure containing the emissive surface data.
        - trainable_bsdf: mi.BSDF. 
        - num_points: int. The number of surface point samples to use.
        - num_wi: int. The number of incident directions per surface point to use to calculate the radiosity integral.
    Outputs:
        - loss: Float. The scalar loss.
    
    # TODO: we can further re-arrange the steps to cut down/consolidate kernel launches
    '''
    # Temp workaround. TODO: avoid initializing a new sampler at each iteration
    sampler: mi.Sampler = mi.load_dict({'type': 'independent'})

    # Sample `NUM_POINTS` different surface points
    si, (delta_emitter_sample, delta_emitter_Li) = scene_sampler.sample(sampler, num_points)

    # Evaluate LHS of balance equation
    lhs = radiance_cache.query_cached_Lo(sampler, si)

    # Evaluate RHS integral
    Li, wi_local, si_flattened = radiance_cache.query_cached_Li(sampler, si, num_wi)
    ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)
    f_io, material_pdf = trainable_bsdf.eval_pdf(ctx, si = si_flattened, wo = wi_local)
    integrand = f_io * dr.detach(Li * dr.rcp(material_pdf))
    rhs = dr.block_reduce(dr.ReduceOp.Add, integrand, block_size = num_wi) / num_wi

    # flat_idxs = dr.tile(dr.arange(UInt, num_points), num_wi)
    # rhs = dr.zeros(mi.Color3f, dr.width(lhs))
    # dr.scatter_reduce(dr.ReduceOp.Add, rhs, integrand, flat_idxs)
    # rhs /= num_wi

    # print(f"term1:{rhs}")

    # Add contribution from scene emitters
    ctx = mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All)
    delta_emitter_Li = dr.detach(delta_emitter_Li)
    delta_emitter_Li *= trainable_bsdf.eval(ctx, si, wo = si.to_local(delta_emitter_sample.d))
    rhs += delta_emitter_Li
    lhs = dr.detach(lhs)

    # # Account for surface emission at `A`
    # lhs += radiance_cache.query_Le(si)

    
    # print(f"term2:{delta_emitter_Li}")

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
