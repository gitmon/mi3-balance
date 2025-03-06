import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
from drjit.auto import Float, UInt

from spherical_harmonics import get_sh_count, get_sh_order_from_index, spherical_integrate

def eval_envmap(envmap: mi.Emitter, d: mi.Vector3f, scale = None, Lmax = None):
    si = dr.zeros(mi.SurfaceInteraction3f, dr.width(d))
    si.wi = -d
    Le = envmap.eval(si)
    if scale is not None:
        Le *= scale
    if Lmax is not None:
        Le = dr.minimum(Le, Lmax)
    return Le

def eval_envmap_on_sphere(envmap: mi.Emitter, N: int, scale = None, Lmax = None):
    sampler = mi.load_dict({'type': 'orthogonal'})
    sampler.seed(0, N)
    points_ref = mi.warp.square_to_uniform_sphere(sampler.next_2d())
    return eval_envmap(envmap, points_ref, scale=scale, Lmax=Lmax), points_ref

def test_spherical_integrate_0(Ns = [16, 32, 64, 128, 256, 512]):
    '''
    Use the spherical integrator compute the surface area of a unit sphere.
    '''
    out = []
    for N in Ns:
        I = spherical_integrate(lambda d: 1.0, N)
        error = I - dr.four_pi
        error = abs(error.numpy().item())
        out.append((N, error))
        print(f"{N}:\t{error:3e}")
    out = np.array(out)

    plt.figure()
    plt.title("Test: sphere surface area")
    plt.loglog(out[:,0], out[:,1], label="numeric")
    plt.loglog(out[:,0], out[:,0] ** -2, label="O(h**2)")
    plt.legend()

def test_spherical_integrate_1(index: int, Ns = [16, 32, 64, 128, 256, 512]):
    '''
    Use the spherical integrator to compute the inner product of a spherical harmonic basis function with itself.
    '''
    out = []
    ord = get_sh_order_from_index(index)
    for N in Ns:
        I = spherical_integrate(lambda d: dr.square(dr.sh_eval(d, ord)[index]), N)
        error = I - 1.0
        error = abs(error.numpy().item())
        out.append((N, error))
        print(f"{N}:\t{error:3e}")
    out = np.array(out)

    plt.figure()
    plt.title(f"Test: SH basis function #{index}")
    plt.loglog(out[:,0], out[:,1], label="numeric")
    plt.loglog(out[:,0], out[:,0] ** -2, label="O(h**2)")
    plt.legend()

def test_spherical_integrate_2(max_order: int = 2, num_points: int = 1024):
    '''
    Use the spherical integrator to compute the inner product of spherical harmonic basis functions with each other.
    The orthonormality property should be satisfied almost exactly.
    '''
    # check orthogonality of SH basis functions
    N = get_sh_count(max_order)
    correlation = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            I = spherical_integrate(lambda d: dr.sh_eval(d, max_order)[i] * dr.sh_eval(d, max_order)[j], num_points)
            correlation[i,j] = I.numpy().item()

    print(f"Max orthogonality error: {np.max(np.abs(correlation - np.eye(N))):.3e}")
    assert np.allclose(correlation, np.eye(N), atol=1e-5)
    return correlation

def test_spherical_integrate_env(index: int, Ns = [16, 32, 64, 128, 256, 512, 1024]):
    '''
    Use the spherical integrator to compute the inner product of a spherical harmonic basis function with an input
    envmap.
    '''
    envmap = mi.load_dict({
        "type": "envmap",
        # "filename": "/home/jonathan/Documents/mi3-balance/resources/data/common/textures/museum.exr"
        "filename": "/home/jonathan/Documents/mi3-balance/scripts/resources/envmaps/rosendal_park_sunset_puresky_1k.exr"
    })

    f = lambda d: eval_envmap(envmap, d, Lmax=5.0).x * dr.sh_eval(d, ord)[index]
    out = []
    ord = get_sh_order_from_index(index)
    I_ref = spherical_integrate(f, Ns[-1] << 3)

    for N in Ns:
        I = spherical_integrate(f, N)
        error = I - I_ref
        error = abs(error.numpy().item())
        out.append((N, error))
        print(f"{N}:\t{error:3e}")
    out = np.array(out)

    plt.figure()
    plt.title(f"Test: <Museum envmap, SH#{index}>")
    plt.loglog(out[:,0], out[:,1], label="numeric")
    plt.loglog(out[:,0], out[:,0] ** -2, label="O(h**2)")
    plt.legend()


from spherical_harmonics import fit_sh_coeffs_color, eval_sh_coeffs_color_on_sphere, eval_sh_coeffs_color_for_direction
import polyscope as ps

def test_fit_sh(max_order: int):
    # Randomly generate a spherical function using the leading SH basis functions
    num_coeffs = get_sh_count(max_order)
    sampler = mi.load_dict({'type':'independent'})
    sampler.seed(0, num_coeffs)
    coeffs_ref = mi.Color3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())

    # Fit the SH coefficients
    N_fit = 1 << 8
    coeffs_fit = fit_sh_coeffs_color(lambda d: eval_sh_coeffs_color_for_direction(coeffs_ref, d), max_order, N_fit)

    # Eval both reference and fitted functions on the unit sphere
    N_eval = 1 << 16
    colors_fit, points_fit = eval_sh_coeffs_color_on_sphere(coeffs_fit, N_eval)
    points_fit += mi.Point3f(0.0, 0.0, 2.0)
    colors_ref, points_ref = eval_sh_coeffs_color_on_sphere(coeffs_ref, N_eval)

    errors   = dr.norm(colors_ref - colors_fit)
    errors_R = dr.abs((colors_ref - colors_fit).x)
    errors_G = dr.abs((colors_ref - colors_fit).y)
    errors_B = dr.abs((colors_ref - colors_fit).z)

    print(f"Mean L2 error (color): {dr.mean(errors)}")
    print(f"Mean L2 error (params): {dr.mean(dr.norm(coeffs_fit - coeffs_ref))}")

    ps.init()
    cloud_ref = ps.register_point_cloud("Reference", points_ref.numpy().T)
    cloud_ref.add_color_quantity("Colors", colors_ref.numpy().T)

    cloud_fit = ps.register_point_cloud("Fitted", points_fit.numpy().T)
    cloud_fit.add_color_quantity("Colors", colors_fit.numpy().T)
    cloud_fit.add_scalar_quantity("ErrorB", errors_B.numpy().T)
    cloud_fit.add_scalar_quantity("ErrorG", errors_G.numpy().T)
    cloud_fit.add_scalar_quantity("ErrorR", errors_R.numpy().T)
    cloud_fit.add_scalar_quantity("Errors", errors.numpy().T)
    ps.show()

def test_fit_envmap(max_order: int):
    home_dir = "/home/jonathan/Documents/mi3-balance/scripts/resources/envmaps/"
    # envmap_fp, scale, Lmax = "rosendal_park_sunset_puresky_1k.exr", 0.1, 5.0
    # envmap_fp, scale, Lmax = "lilienstein_1k.exr", 1.0, 5.0
    # envmap_fp, scale, Lmax = "lonely_road_afternoon_puresky_1k.exr", 1.0, 2.0
    envmap_fp, scale, Lmax = "rural_asphalt_road_1k.exr", 1.0, 2.0
    envmap = mi.load_dict({'type': 'envmap', 'filename': home_dir + envmap_fp})

    # Compute SH coefficients
    N_fit = 1 << 8
    coeffs = fit_sh_coeffs_color(lambda d: eval_envmap(envmap, d, scale=scale, Lmax=Lmax), max_order, N_fit)

    # Eval both reference and fitted functions on the unit sphere
    N_eval = 1 << 16
    colors_fit, points_fit = eval_sh_coeffs_color_on_sphere(coeffs, N_eval)
    points_fit += mi.Point3f(0.0, 0.0, 2.0)
    colors_ref, points_ref = eval_envmap_on_sphere(envmap, N_eval, scale, Lmax)

    errors   = dr.norm(colors_ref - colors_fit)
    errors_R = dr.abs((colors_ref - colors_fit).x)
    errors_G = dr.abs((colors_ref - colors_fit).y)
    errors_B = dr.abs((colors_ref - colors_fit).z)

    print(f"Mean L2 error (color): {dr.mean(errors)}")

    ps.init()
    cloud_ref = ps.register_point_cloud("Reference", points_ref.numpy().T)
    cloud_ref.add_color_quantity("Colors", colors_ref.numpy().T)

    cloud_fit = ps.register_point_cloud("Fitted", points_fit.numpy().T)
    cloud_fit.add_color_quantity("Colors", colors_fit.numpy().T)
    cloud_fit.add_scalar_quantity("ErrorB", errors_B.numpy().T)
    cloud_fit.add_scalar_quantity("ErrorG", errors_G.numpy().T)
    cloud_fit.add_scalar_quantity("ErrorR", errors_R.numpy().T)
    cloud_fit.add_scalar_quantity("Errors", errors.numpy().T)
    ps.show()

if __name__ == "__main__":
    test_spherical_integrate_0()
    test_spherical_integrate_1(0)
    test_spherical_integrate_1(1)
    test_spherical_integrate_1(3)
    test_spherical_integrate_1(5)
    test_spherical_integrate_1(8)
    test_spherical_integrate_env(8)
    test_spherical_integrate_2(3)
