import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-balance/build/python')

import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt

# High level organization. The loss computation is split into three kernels:
# - Sample geometry points and Lo ray directions
# - Compute Lo
# - Compute Li

class SceneSurfaceSampler:
    def __init__(self, scene: mi.Scene):
        shape_ptrs = scene.shapes_dr()
        # TODO: can't figure out how to concatenate the areas (== dr.Float of size [1,])
        areas = [shape.surface_area().numpy().item() for shape in shape_ptrs]
        areas_dr = Float(areas)
        self.distribution = mi.DiscreteDistribution(areas_dr)
        self.shape_ptrs = shape_ptrs

    def sample(self, sampler: mi.Sampler, num_points: int) -> mi.SurfaceInteraction3f:
        # Generate `NUM_POINTS` different surface samples
        # TODO: the seed value matters!
        sampler.seed(0, num_points)
        idx = self.distribution.sample(sampler.next_1d(), True)
        shape = dr.gather(mi.ShapePtr, self.shape_ptrs, idx)
        si = shape.eval_parameterization(sampler.next_2d(), active=True)
        # Generate one outgoing ray per surface sample
        wo_local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
        si.wi = wo_local
        return si
    
class RadianceCacheMITSUBA:
    def __init__(self, scene: mi.Scene):
        self.scene = scene
        self.integrator = scene.integrator()

    def query_cached_Lo(self, sampler: mi.Sampler, si: mi.SurfaceInteraction3f, spp: int)  -> mi.Color3f:
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
        ray_flat_idxs = dr.repeat(dr.arange(UInt, num_points), spp)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, Lo_rays, ray_flat_idxs, mode = dr.ReduceMode.Direct)
        sampler.seed(0, num_points * spp)
        colors, _, _ = self.integrator.sample(self.scene, sampler, rays_flattened)
        Lo = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = spp) / spp
        # TODO: why is there this missing factor, where does it come from?
        Lo *= si.bsdf().eval_diffuse_reflectance(si, True)
        return Lo

    def query_cached_Li(self, sampler: mi.Sampler, si: mi.SurfaceInteraction3f, num_wi: int, spp_per_wi: int) -> mi.Color3f:
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
        sampler.seed(0, num_points * num_wi)
        # the `flat_idxs` has the form: 
        # [0, ..., 0, 1, ..., 1,    ...    N-1, ..., N-1]   (contiguous order)
        si_flat_idxs = dr.repeat(dr.arange(UInt, num_points), num_wi)
        # `si_flattened` has the form:
        # [s0, ..., s0, s1, ..., s1,    ...    sN-1, ..., sN-1]
        # NOTE: `dr.ReduceMode` is not used for the `gather()` itself, but instead to 
        # implement its adjoint operation (scatter) for reverse-mode AD. For this context,
        # the choice doesn't matter since we never backprop through this part of the algorithm,
        # but `ReduceMode.Local` is probably optimal when our arrays are laid out 
        # contiguously, as we do here.
        si_flattened = dr.gather(mi.SurfaceInteraction3f, si, si_flat_idxs, mode = dr.ReduceMode.Local)

        # Compute the outgoing radiance from `A` for a direction, `wi`
        # Note that `wo` is stored in the `si.wi` field (unintuitive, but needed for BSDF.eval() later)
        wi_local = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
        wi_world = si_flattened.to_world(wi_local)
        wi_rays = si_flattened.spawn_ray(wi_world)

        # Compute Li for each of the incident directions. For each `Li_ray`, trace `SPP_LI` 
        # different MC samples and average them to get the outgoing radiance.
        NUM_RAYS = num_points * num_wi
        ray_flat_idxs = dr.repeat(dr.arange(UInt, NUM_RAYS), spp_per_wi)    # contiguous order
        rays_flattened = dr.gather(mi.Ray3f, wi_rays, ray_flat_idxs, mode = dr.ReduceMode.Local)
        sampler.seed(0, NUM_RAYS * spp_per_wi)
        colors, _, _ = self.integrator.sample(self.scene, sampler, rays_flattened)
        Li = dr.block_reduce(dr.ReduceOp.Add, colors, block_size = spp_per_wi) / spp_per_wi
        return Li, wi_local, si_flattened

def compute_loss(
        scene_sampler: SceneSurfaceSampler, 
        radiance_cache: RadianceCacheMITSUBA, 
        trainable_bsdf: mi.BSDF, 
        num_points: int,
        num_wi: int,
        spp_per_wo: int,
        spp_per_wi: int):
    sampler = mi.load_dict({'type': 'independent'})
    # Sample `NUM_POINTS` different surface points
    si = scene_sampler.sample(sampler, num_points)

    # Evaluate LHS of balance equation
    lhs = radiance_cache.query_cached_Lo(sampler, si, spp_per_wo)

    # Evaluate RHS integral
    Li, wi_local, si_flattened = radiance_cache.query_cached_Li(sampler, si, num_wi, spp_per_wi)
    f_io, pdf = trainable_bsdf.eval_pdf(
                    mi.BSDFContext(mi.TransportMode.Radiance, mi.BSDFFlags.All), 
                    si = si_flattened, wo = wi_local)
    rhs = dr.mean(f_io * dr.detach(Li * dr.rcp(pdf)), axis=1)
    lhs = dr.detach(lhs)
    return 0.5 * dr.sum(dr.squared_norm(lhs - rhs)) / num_points
