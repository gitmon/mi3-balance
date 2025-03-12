import drjit as dr
import mitsuba as mi
import numpy as np
from drjit.auto import Float, UInt, Bool
from principled_bsdf import Principled

M_BASE_COLOR = [0.2, 0.3, 0.9]
M_ROUGHNESS = 0.2
M_METALLIC = 0.7
M_ANISOTROPIC = 0.7
M_SPEC_TINT = 0.2
M_SPECULAR = 0.6

def generate_geo() -> None:
    '''
    Generate a tessellated rectangle of size (-1,1) x (-1,1) on the XY plane.
    '''
    from gpytoolbox import regular_square_mesh, write_mesh
    import numpy as np

    V_, F = regular_square_mesh(2, 2)
    Nv = V_.shape[0]
    V = np.c_[V_[:,0], V_[:,1], np.zeros(Nv)]
    write_mesh('./resources/rectangle.obj', V, F)

def make_scene(
        m_base_color: list[float],
        m_roughness: float,
        m_metallic: float,
        m_anisotropic: float,
        m_spec_tint: float,
        m_specular: float,
    ) -> mi.Scene:
    '''
    Define the test scene.
    '''
    generate_geo()
    
    base_color = [0.2, 0.3, 0.9]

    bsdf_dict = {
        'type': 'principled',
        'base_color': {
            'type': 'rgb',
            'value': m_base_color,
        },
        'roughness': m_roughness,
        'metallic': m_metallic,
        'anisotropic': m_anisotropic,
        # 'spec_tint': m_spec_tint,
        'specular': m_specular,
    }

    scene_dict = {
        'type': 'scene',
        'myintegrator': {
            'type': 'path',
            'max_depth': 6,
        },
        'geo': {
            'type': 'obj',
            'filename': './resources/rectangle.obj',
            'bsdf': bsdf_dict,
            },
        'emitter': {
            'type': 'point',
            'position': [0.0, 0.0, 2.0],
            'intensity': {
                'type': 'uniform',
                'value': 10.0,
            }
        },
        'sensor': {
            'type': 'perspective',
            'fov': 90,
            'to_world': mi.ScalarTransform4f().look_at(
                origin = mi.ScalarPoint3f([0.0, 0.0, 1.0]),
                target = mi.ScalarPoint3f([0.0, 0.0, 0.0]),
                up     = mi.ScalarPoint3f([0.0, 1.0, 0.0])),
            "film": {
                "type": "hdrfilm",
                "width": 256,
                "height": 256,
            },
            "sampler": {
                "type": "independent",
                "sample_count": 64,
            }
        }
    }
    return mi.load_dict(scene_dict)

def test_principled(num_samples = 1 << 20, random_inputs: bool = False):
    if random_inputs:
        m_base_color = np.random.rand(3)
        m_roughness = np.random.rand()
        m_metallic = np.random.rand()
        m_anisotropic = np.random.rand()
        m_spec_tint = np.random.rand()
        m_specular = np.random.rand()
    else:
        m_base_color = M_BASE_COLOR
        m_roughness = M_ROUGHNESS
        m_metallic = M_METALLIC
        m_anisotropic = M_ANISOTROPIC
        m_spec_tint = M_SPEC_TINT
        m_specular = M_SPECULAR

    scene = make_scene(
        m_base_color,
        m_roughness,
        m_metallic,
        m_anisotropic,
        m_spec_tint,
        m_specular)
    
    # Create a surface interaction
    ray, _ = scene.sensors()[0].sample_ray(0.0, 0.0, mi.Point2f(0.5, 0.5), dr.zeros(mi.Point2f))
    si_init = scene.ray_intersect(ray)

    # Initialize BSDF and populate the mesh attributes
    mesh = scene.shapes()[0]
    bsdf_ref = scene.shapes()[0].bsdf()
    bsdf_impl = Principled(
        has_metallic = True, 
        has_anisotropic = True, 
        has_spec_tint = False, 
        specular = m_specular)
    bsdf_impl.initialize_mesh_attributes(
        mesh, 
        m_base_color,
        m_roughness,
        m_metallic,
        m_anisotropic,
        m_spec_tint)

    # Generate input for BSDFs at `si`
    sampler = mi.load_dict({ 'type': 'independent' })
    sampler.seed(0, num_samples)

    si = dr.gather(mi.SurfaceInteraction3f, si_init, dr.zeros(UInt, num_samples))
    si.wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
    wo = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
    sample1, sample2 = sampler.next_1d(), sampler.next_2d()

    # Define error norms
    def MaxNorm(a, b):
        max_value = dr.maximum(dr.max(a, axis=None), dr.max(b, axis=None))
        return (dr.max(dr.abs(a - b), axis=None) / max_value).numpy().item()

    def L2Norm(a, b):
        normsq_a = dr.mean(dr.squared_norm(a))
        normsq_b = dr.mean(dr.squared_norm(b))
        max_value = dr.maximum(dr.sqrt(normsq_a), dr.norm(normsq_b))
        return (dr.sqrt(dr.mean(dr.squared_norm(a - b))) / max_value).numpy().item()

    # Evaluate reference BSDF implementation against our own
    ALL_ACTIVE = dr.full(Bool, 1, num_samples)
    ctx = mi.BSDFContext()

    eval_ref = bsdf_ref.eval(ctx, si, wo)
    eval_impl = bsdf_impl.eval(si, wo, ALL_ACTIVE)

    bs_ref, sampled_pdf_ref = bsdf_ref.sample(ctx, si, sample1, sample2)
    bs_impl, sampled_pdf_impl = bsdf_impl.sample(si, sample1, sample2, ALL_ACTIVE)

    eval_pdf_ref = bsdf_ref.pdf(ctx, si, wo)
    eval_pdf_impl = bsdf_impl.pdf(si, wo, ALL_ACTIVE)

    # Evaluate error
    print("\t\terr_Linf\terr_L2")
    print(f"Eval:\t\t{MaxNorm(eval_ref, eval_impl):.3e}" + \
        f"\t{L2Norm(eval_ref, eval_impl):.3e}")
    print(f"Pdf:\t\t{MaxNorm(eval_pdf_ref, eval_pdf_impl):.3e}" + \
        f"\t{L2Norm(eval_pdf_ref, eval_pdf_impl):.3e}")
    print(f"SamplePdf:\t{MaxNorm(sampled_pdf_ref, sampled_pdf_impl):.3e}" + \
        f"\t{L2Norm(sampled_pdf_ref, sampled_pdf_impl):.3e}")

    print("SampleRec:")
    print(f"\two\t{MaxNorm(bs_ref.wo, bs_impl.wo):.3e}\t{L2Norm(bs_ref.wo, bs_impl.wo):.3e}")
    print(f"\tpdf\t{MaxNorm(bs_ref.pdf, bs_impl.pdf):.3e}\t{L2Norm(bs_ref.pdf, bs_impl.pdf):.3e}")
    print(f"\teta\t{MaxNorm(bs_ref.eta, bs_impl.eta):.3e}\t{L2Norm(bs_ref.eta, bs_impl.eta):.3e}")
    print(f"\ttype\t{dr.allclose(bs_ref.sampled_type, bs_impl.sampled_type)}" + \
        f"\t\t{dr.allclose(bs_ref.sampled_type, bs_impl.sampled_type)}")
    print(f"\tcomp\t{dr.allclose(bs_ref.sampled_component, bs_impl.sampled_component)}" + \
        f"\t\t{dr.allclose(bs_ref.sampled_component, bs_impl.sampled_component)}")
    
    return scene

import matplotlib.pyplot as plt

def plot_principled(Nx = 1 << 10, random_inputs: bool = False):
    if random_inputs:
        m_base_color = np.random.rand(3)
        m_roughness = np.random.rand()
        m_metallic = np.random.rand()
        m_anisotropic = np.random.rand()
        m_spec_tint = np.random.rand()
        m_specular = np.random.rand()
    else:
        m_base_color = M_BASE_COLOR
        m_roughness = M_ROUGHNESS
        m_metallic = M_METALLIC
        m_anisotropic = M_ANISOTROPIC
        m_spec_tint = M_SPEC_TINT
        m_specular = M_SPECULAR

    scene = make_scene(
        m_base_color,
        m_roughness,
        m_metallic,
        m_anisotropic,
        m_spec_tint,
        m_specular)
    
    # Create a surface interaction
    ray, _ = scene.sensors()[0].sample_ray(0.0, 0.0, mi.Point2f(0.5, 0.5), dr.zeros(mi.Point2f))
    si_init = scene.ray_intersect(ray)

    # Initialize BSDF and populate the mesh attributes
    mesh = scene.shapes()[0]
    bsdf_ref = scene.shapes()[0].bsdf()
    bsdf_impl = Principled(
        has_metallic = True, 
        has_anisotropic = True, 
        has_spec_tint = False, 
        specular = m_specular)
    bsdf_impl.initialize_mesh_attributes(
        mesh, 
        m_base_color,
        m_roughness,
        m_metallic,
        m_anisotropic,
        m_spec_tint)

    # Generate input for BSDFs at `si`
    num_samples = 2 * Nx * Nx
    sampler = mi.load_dict({ 'type': 'independent' })
    si_init.wi = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())
    sampler.seed(0, num_samples)

    si = dr.gather(mi.SurfaceInteraction3f, si_init, dr.zeros(UInt, num_samples))
    us, vs = dr.meshgrid(
        dr.linspace(Float, 0.0, 1.0, Nx), 
        dr.linspace(Float, 0.0, 1.0, 2 * Nx))

    wo = mi.warp.square_to_uniform_hemisphere(mi.Point2f(us, vs))

    # Evaluate reference BSDF implementation against our own
    ALL_ACTIVE = dr.full(Bool, 1, num_samples)
    ctx = mi.BSDFContext()

    eval_ref = bsdf_ref.eval(ctx, si, wo)
    eval_impl = bsdf_impl.eval(si, wo, ALL_ACTIVE)

    eval_pdf_ref = bsdf_ref.pdf(ctx, si, wo)
    eval_pdf_impl = bsdf_impl.pdf(si, wo, ALL_ACTIVE)

    # Evaluate error
    eval_ref  = mi.TensorXf(dr.norm(eval_ref), (Nx, 2 * Nx))
    eval_impl = mi.TensorXf(dr.norm(eval_impl), (Nx, 2 * Nx))
    plt.figure(figsize=(8,12), dpi=200)
    plt.subplot(311); plt.title(r"$f_r(\omega_{o}, \cdot)$, ref"); plt.imshow(eval_ref); plt.colorbar()
    plt.subplot(312); plt.title(r"$f_r(\omega_{o}, \cdot)$, new"); plt.imshow(eval_impl); plt.colorbar()
    plt.subplot(313); plt.title(r"$e(f_r(\omega_{o}, \cdot))$"); plt.imshow(dr.abs(eval_ref - eval_impl)); plt.colorbar()
    plt.tight_layout()

    eval_pdf_ref  = mi.TensorXf(eval_pdf_ref, (Nx, 2 * Nx))
    eval_pdf_impl = mi.TensorXf(eval_pdf_impl, (Nx, 2 * Nx))
    plt.figure(figsize=(8,12), dpi=200)
    plt.subplot(311); plt.title(r"$pdf(\omega_{o}, \cdot)$, ref"); plt.imshow(eval_pdf_ref); plt.colorbar()
    plt.subplot(312); plt.title(r"$pdf(\omega_{o}, \cdot)$, new"); plt.imshow(eval_pdf_impl); plt.colorbar()
    plt.subplot(313); plt.title(r"$e(f_r(\omega_{o}, \cdot))$"); plt.imshow(dr.abs(eval_pdf_ref - eval_pdf_impl)); plt.colorbar()
    plt.tight_layout()


if __name__ == '__main__':
    test_principled()