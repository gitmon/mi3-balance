import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool

### Helper functions from `principledhelpers.h`

def mulsign_neg(a, b):
    # mulsign_neg(x) should be equiv. to -mulsign(x)
    return -dr.mulsign(a, b)

def fnmadd(a, b, c):
    # `fnmadd(a,b,c)` == `-a * b + c`, which can be implemented as either
    # `fmadd(-a,b,c)` or `fmadd(a,-b,c)`
    return dr.fma(-a, b, c)

def calc_dist_params(anisotropic: Float, roughness: Float, has_anisotropic: bool) -> tuple[Float, Float]:
    roughness_2 = dr.square(roughness)
    if not(has_anisotropic):
        a = dr.maximum(0.001, roughness_2)
        return (a, a)
    aspect = dr.sqrt(1.0 - 0.9 * anisotropic)
    return ( dr.maximum(0.001, roughness_2 / aspect),
             dr.maximum(0.001, roughness_2 * aspect))

def mac_mic_compatibility(m: mi.Vector3f, wi: mi.Vector3f, wo: mi.Vector3f, cos_theta_i: Float, reflection: bool) -> Bool:
    # NOTE: elementwise AND (this syntax should be fine)
    if reflection:
        return (dr.dot(wi, dr.mulsign(m, cos_theta_i)) > 0.0) & \
            (dr.dot(wo, dr.mulsign(m, cos_theta_i)) > 0.0)
    else:
        return (dr.dot(wi, dr.mulsign(m, cos_theta_i)) > 0.0) & \
            (dr.dot(wo, mulsign_neg(m, cos_theta_i)) > 0.0)

def schlick_R0_eta(eta: Float) -> Float:
    return dr.square((eta - 1.0) / (eta + 1.0))

def schlick_weight(cos_i: Float) -> Float:
    m = dr.clip(1.0 - cos_i, 0.0, 1.0)
    return dr.square(dr.square(m)) * m

def calc_schlick(R0: mi.Color3f, cos_theta_i: Float, eta: Float) -> mi.Color3f:
    outside_mask = cos_theta_i >= 0.0
    rcp_eta = dr.rcp(eta)
    eta_it  = dr.select(outside_mask, eta, rcp_eta)
    eta_ti  = dr.select(outside_mask, rcp_eta, eta)

    cos_theta_t_sqr = fnmadd(
            fnmadd(cos_theta_i, cos_theta_i, 1.0), dr.square(eta_ti), 1.0)
    cos_theta_t = dr.safe_sqrt(cos_theta_t_sqr)
    return dr.select(
            eta_it > 1.0,
            dr.lerp(schlick_weight(dr.abs(cos_theta_i)), 1.0, R0),
            dr.lerp(schlick_weight(cos_theta_t), 1.0, R0))

def principled_fresnel(
        F_dielectric: Float, 
        metallic: Float,
        spec_tint: Float,
        base_color: mi.Color3f,
        lum: Float, cos_theta_i: Float,
        front_side: Bool,
        bsdf: Float, eta: Float,
        has_metallic: bool, has_spec_tint: bool) -> mi.Color3f:
    # Outside mask based on micro surface
    outside_mask = cos_theta_i >= 0.0
    rcp_eta = dr.rcp(eta)
    eta_it  = dr.select(outside_mask, eta, rcp_eta)
    # TODO: does the width need to be set?
    F_schlick = mi.Color3f(0.0) 

    # Metallic component based on Schlick.
    if has_metallic:
        F_schlick += metallic * calc_schlick(base_color, cos_theta_i, eta)

    # Tinted dielectric component based on Schlick.
    if has_spec_tint:
        c_tint = dr.select(lum > 0.0, base_color / lum, 1.0)
        F0_spec_tint = c_tint * schlick_R0_eta(eta_it)
        F_schlick += \
                (1.0 - metallic) * spec_tint * \
                calc_schlick(F0_spec_tint, cos_theta_i, eta)

    # Front side fresnel
    F_front = (1.0 - metallic) * (1.0 - spec_tint) * F_dielectric + F_schlick
    # For back side there is no tint or metallic, just true dielectric
    # fresnel.
    return dr.select(front_side, F_front, bsdf * F_dielectric)


### Mesh attribute query functions

def eval_base_color(si: mi.SurfaceInteraction3f, active: Bool) -> mi.Color3f:
    return si.shape.eval_attribute_3("vertex_bsdf_base_color", si, active)

def eval_roughness(si: mi.SurfaceInteraction3f, active: Bool) -> Float:
    return si.shape.eval_attribute_1("vertex_bsdf_roughness", si, active)

def eval_metallic(si: mi.SurfaceInteraction3f, active: Bool) -> Float:
    return si.shape.eval_attribute_1("vertex_bsdf_metallic", si, active)

def eval_anisotropic(si: mi.SurfaceInteraction3f, active: Bool) -> Float:
    return si.shape.eval_attribute_1("vertex_bsdf_anisotropic", si, active)

def eval_spec_tint(si: mi.SurfaceInteraction3f, active: Bool) -> Float:
    return si.shape.eval_attribute_1("vertex_bsdf_spec_tint", si, active)


### Method implementations for principled BSDF

def bsdf_eval(
            si: mi.SurfaceInteraction3f, 
            wo: mi.Vector3f, 
            active: Bool,
            # class members
            m_specular: float,
            m_has_anisotropic: bool,
            m_has_metallic: bool,
            m_has_spec_tint: bool,
            ) -> mi.Color3f:
    m_eta = 2.0 * dr.rcp(1.0 - dr.sqrt(0.08 * m_specular)) - 1.0

    # TODO: should invocations of `Frame3f.cos_theta()` be using the shading frame of the `si` instead?
    cos_theta_i = mi.Frame3f.cos_theta(si.wi)
    # Ignore perfectly grazing configurations
    active &= cos_theta_i != 0.0

    # if (unlikely(dr::none_or<false>(active)))
    if dr.none(active):
        return 0.0

    # Store the weights.
    mesh = si.shape
    active &= mesh.is_mesh()
    # assert mesh.is_mesh(), "`si` is not on a mesh!"
    anisotropic = eval_anisotropic(si, active) if m_has_anisotropic else 0.0
    roughness = eval_roughness(si, active)
    metallic = eval_metallic(si, active) if m_has_metallic else 0.0
    base_color = eval_base_color(si, active)

    # Weights for BRDF and BSDF major lobes.
    brdf = (1.0 - metallic)
    bsdf = 0.0

    cos_theta_o = mi.Frame3f.cos_theta(wo)

    # Reflection and refraction masks.
    reflect = cos_theta_i * cos_theta_o > 0.0
    refract = cos_theta_i * cos_theta_o < 0.0

    # Masks for the side of the incident ray (wi.z<0)
    front_side = cos_theta_i > 0.0
    inv_eta   = dr.rcp(m_eta)

    # Eta value w.r.t. ray instead of the object.
    eta_path     = dr.select(front_side, m_eta, inv_eta)
    # inv_eta_path = dr.select(front_side, inv_eta, m_eta)

    # Main specular reflection and transmission lobe

    ax, ay = calc_dist_params(anisotropic, roughness, m_has_anisotropic)
    spec_dist = mi.MicrofacetDistribution(
           mi.MicrofacetType.GGX, 
           alpha_u = ax, alpha_v = ay)

    # Halfway vector
    wh = dr.normalize(si.wi + wo * dr.select(reflect, 1.0, eta_path))

    # Make sure that the halfway vector points outwards the object
    wh = dr.mulsign(wh, mi.Frame3f.cos_theta(wh))

    # Dielectric Fresnel
    F_spec_dielectric, cos_theta_t, eta_it, eta_ti = mi.fresnel(dr.dot(si.wi, wh), m_eta)

    reflection_compatibilty = \
        mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, True)

    # Masks for evaluating the lobes.
    # Specular reflection mask
    spec_reflect_active = active & reflect & \
        reflection_compatibilty & \
        (F_spec_dielectric > 0.0)
    
    # Diffuse, retro and fake subsurface mask
    diffuse_active = active & (brdf > 0.0) & reflect & front_side
    # Evaluate the microfacet normal distribution
    D = spec_dist.eval(wh)
    # Smith's shadowing-masking function
    G = spec_dist.G(si.wi, wo, wh)
    # Initialize the final BSDF value.
    value = dr.zeros(mi.Color3f)
    # Main specular reflection evaluation
    # if (dr::any_or<true>(spec_reflect_active)) {
    if dr.any(spec_reflect_active):
        # No need to calculate luminance if there is no color tint.
        lum = mi.luminance(base_color, si.wavelengths) if m_has_spec_tint else 1.0
        spec_tint = eval_spec_tint(si, active) if m_has_spec_tint else 0.0
        # Fresnel term
        F_principled = principled_fresnel(
            F_spec_dielectric, metallic, spec_tint, base_color, lum,
            dr.dot(si.wi, wh), front_side, bsdf, m_eta, m_has_metallic,
            m_has_spec_tint)
        # Adding the specular reflection component
        value = dr.select(spec_reflect_active, value + \
                F_principled * D * G / (4.0 * dr.abs(cos_theta_i)), \
                value)

    # Evaluation of diffuse, retro reflection, fake subsurface and
    # sheen.
    # if (dr::any_or<true>(diffuse_active)) {
    if dr.any(diffuse_active):
        Fo = schlick_weight(dr.abs(cos_theta_o))
        Fi = schlick_weight(dr.abs(cos_theta_i))
        # Diffuse
        f_diff = (1.0 - 0.5 * Fi) * (1.0 - 0.5 * Fo)
        cos_theta_d = dr.dot(wh, wo)
        Rr          = 2.0 * roughness * dr.square(cos_theta_d)
        # Retro reflection
        f_retro = Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0))
        # Adding diffuse, retro evaluation. (no fake ss.)
        value = dr.select(diffuse_active, value + \
            brdf * dr.abs(cos_theta_o) * base_color * \
            dr.inv_pi * (f_diff + f_retro), \
            value)
    return value & active

def bsdf_pdf(
            si: mi.SurfaceInteraction3f, 
            wo: mi.Vector3f, 
            active: Bool,
            # class members
            m_specular: float,
            m_has_anisotropic: bool,
            m_has_metallic: bool,
            m_spec_srate: float = 1.0,
            m_diff_refl_srate: float = 1.0,
            ) -> Float:
    m_eta = 2.0 * dr.rcp(1.0 - dr.sqrt(0.08 * m_specular)) - 1.0

    cos_theta_i = mi.Frame3f.cos_theta(si.wi)
    # Ignore perfectly grazing configurations.
    active &= cos_theta_i != 0.0

    # if (unlikely(dr::none_or<false>(active)))
    if dr.none(active):
        return 0.0

    # Store the weights.
    mesh = si.shape
    active &= mesh.is_mesh()
    # assert mesh.is_mesh(), "`si` is not on a mesh!"
    anisotropic = eval_anisotropic(si, active) if m_has_anisotropic else 0.0
    roughness = eval_roughness(si, active)
    metallic = eval_metallic(si, active) if m_has_metallic else 0.0

    # BRDF and BSDF major lobe weights
    brdf = 1.0 - metallic

    # Masks if incident direction is inside (wi.z<0)
    front_side = cos_theta_i > 0.0

    # Eta w.r.t. light path.
    eta_path    = dr.select(front_side, m_eta, dr.rcp(m_eta))
    cos_theta_o = mi.Frame3f.cos_theta(wo)

    reflect = cos_theta_i * cos_theta_o > 0.0

    # Halfway vector calculation
    wh = dr.normalize(
        si.wi + wo * dr.select(reflect, Float(1.0), eta_path))

    # Make sure that the halfway vector points outwards the object
    wh = dr.mulsign(wh, mi.Frame3f.cos_theta(wh))

    # Main specular distribution for reflection and transmission.
    ax, ay = calc_dist_params(anisotropic, roughness, m_has_anisotropic)
    spec_dist = mi.MicrofacetDistribution(
        mi.MicrofacetType.GGX, 
        alpha_u = ax, alpha_v = ay)

    # Dielectric Fresnel calculation
    F_spec_dielectric, cos_theta_t, eta_it, eta_ti = \
        mi.fresnel(dr.dot(si.wi, wh), m_eta)

    # Defining the probabilities
    prob_spec_reflect = dr.select(
        front_side,
        m_spec_srate,
        F_spec_dielectric)
    prob_diffuse = dr.select(front_side, brdf * m_diff_refl_srate, 0.)

    # Normalizing the probabilities.
    rcp_tot_prob = dr.rcp(prob_spec_reflect + prob_diffuse)
    prob_spec_reflect *= rcp_tot_prob
    prob_diffuse *= rcp_tot_prob

    # Calculation of dwh/dwo term. Different for reflection and
    # transmission.
    dwh_dwo_abs = dr.abs(dr.rcp(4.0 * dr.dot(wo, wh)))

    # Initializing the final pdf value.
    pdf = Float(0.0)

    # Macro-micro surface compatibility mask for reflection.
    mfacet_reflect_macmic = \
        mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, True) & reflect

    # Adding main specular reflection pdf
    pdf = dr.select(mfacet_reflect_macmic, pdf + \
        prob_spec_reflect * \
        spec_dist.pdf(dr.mulsign(si.wi, cos_theta_i), wh) * dwh_dwo_abs, \
        pdf)
    # Adding cosine hemisphere reflection pdf
    pdf = dr.select(reflect, pdf + \
        prob_diffuse * mi.warp.square_to_cosine_hemisphere_pdf(wo), \
        pdf)

    return pdf

def bsdf_sample(
        si: mi.SurfaceInteraction3f, 
        sample1: Float, 
        sample2: mi.Point2f, 
        active: Bool,
        # class members
        m_specular: float,
        m_has_anisotropic: bool,
        m_has_metallic: bool,
        m_has_spec_tint: bool,
        m_spec_srate: float = 1.0,
        m_diff_refl_srate: float = 1.0,
        ) -> tuple[mi.BSDFSample3f, mi.Color3f]:

    m_eta = 2.0 * dr.rcp(1.0 - dr.sqrt(0.08 * m_specular)) - 1.0

    cos_theta_i = mi.Frame3f.cos_theta(si.wi)
    bs = dr.zeros(mi.BSDFSample3f)

    # Ignoring perfectly grazing incoming rays
    active &= cos_theta_i != 0.0

    # if unlikely(dr::none_or<false>(active)):
    if dr.none(active):
        return (bs, dr.zeros(mi.Color3f))

    # Store the weights.
    mesh = si.shape
    active &= mesh.is_mesh()
    # assert mesh.is_mesh(), "`si` is not on a mesh!"
    anisotropic = eval_anisotropic(si, active) if m_has_anisotropic else 0.0
    roughness = eval_roughness(si, active)
    metallic = eval_metallic(si, active) if m_has_metallic else 0.0

    # Weights of BSDF and BRDF major lobes
    brdf = 1.0 - metallic

    # Mask for incident side. (wi.z<0)
    front_side = cos_theta_i > 0.0

    # Defining main specular reflection distribution
    ax, ay = calc_dist_params(anisotropic, roughness, m_has_anisotropic)
    spec_distr = mi.MicrofacetDistribution(
        mi.MicrofacetType.GGX, 
        alpha_u = ax, alpha_v = ay)

    m_spec = spec_distr.sample(dr.mulsign(si.wi, cos_theta_i), sample2)[0]  # Normal3f

    # Fresnel coefficient for the main specular.
    F_spec_dielectric, cos_theta_t, eta_it, eta_ti = \
        mi.fresnel(dr.dot(si.wi, m_spec), m_eta)

    # If BSDF major lobe is turned off, we do not sample the inside
    # case.
    active &= front_side

    # Probability definitions
    # Inside  the material, just microfacet Reflection and
    # microfacet Transmission is sampled.
    prob_spec_reflect = dr.select(
        front_side,
        m_spec_srate,
        F_spec_dielectric)
    prob_diffuse = dr.select(front_side, brdf * m_diff_refl_srate, 0.0)

    # Normalizing the probabilities.
    prob_diffuse *= dr.rcp(prob_spec_reflect + prob_diffuse)

    # Sampling mask definitions
    curr_prob = Float(0.0)
    sample_diffuse = active & (sample1 < prob_diffuse)
    curr_prob += prob_diffuse
    sample_spec_reflect = active & (sample1 >= curr_prob)

    # Eta will be changed in transmission.
    bs.eta = 1.0

    # Main specular reflection sampling
    # if dr::any_or<true>(sample_spec_reflect)) {
    if dr.any(sample_spec_reflect):
        wo = mi.reflect(si.wi, m_spec)
        # dr.masked(bs.wo, sample_spec_reflect) = wo
        # dr.masked(bs.sampled_component, sample_spec_reflect) = 3
        # dr.masked(bs.sampled_type, sample_spec_reflect) = +mi.BSDFFlags.GlossyReflection
        bs.wo = dr.select(sample_spec_reflect, wo, bs.wo)
        bs.sampled_component = dr.select(sample_spec_reflect, 3, bs.sampled_component)
        bs.sampled_type = dr.select(sample_spec_reflect, +mi.BSDFFlags.GlossyReflection, bs.sampled_type)

        # Filter the cases where macro and micro surfaces do not agree
        # on the same side and reflection is not successful
        reflect = cos_theta_i * mi.Frame3f.cos_theta(wo) > 0.0
        active &= (~sample_spec_reflect |
            (mac_mic_compatibility(mi.Vector3f(m_spec),
                                    si.wi, wo, cos_theta_i, True) &
            reflect))

    # Cosine hemisphere reflection sampling
    # if (dr::any_or<true>(sample_diffuse)) {
    if dr.any(sample_diffuse):
        wo = mi.warp.square_to_cosine_hemisphere(sample2)
        # dr.masked(bs.wo, sample_diffuse)                = wo
        # dr.masked(bs.sampled_component, sample_diffuse) = 0
        # dr.masked(bs.sampled_type, sample_diffuse) = mi.BSDFFlags.DiffuseReflection
        bs.wo = dr.select(sample_diffuse, wo, bs.wo)
        bs.sampled_component = dr.select(sample_diffuse, 0, bs.sampled_component)
        bs.sampled_type = dr.select(sample_diffuse, +mi.BSDFFlags.DiffuseReflection, bs.sampled_type)
        reflect = cos_theta_i * mi.Frame3f.cos_theta(wo) > 0.0
        active &= ~sample_diffuse | reflect

    bs.pdf = bsdf_pdf(si, bs.wo, active, m_specular, m_has_anisotropic, m_has_metallic, m_spec_srate, m_diff_refl_srate)
    active &= bs.pdf > 0.0
    result = bsdf_eval(si, bs.wo, active, m_specular, m_has_anisotropic, m_has_metallic, m_has_spec_tint)
    return (bs, result / bs.pdf & active)


# Main BSDF class

class Principled:
    '''
    Duplicate implementation of the principled BSDF which allows material parameters to be queried from the 
    _vertex attributes_ of a mesh, rather than via texture lookups. This is needed to render and optimize
    materials on meshes don't have a global UV parameterization, e.g. those which are produced from 
    automated/algorithmic generation pipelines.
    '''
    def __init__(self, 
                 has_metallic: bool = True, 
                 has_anisotropic: bool = False, 
                 has_spec_tint: bool = False, 
                 specular: float = 0.5):
        self.has_anisotropic = has_anisotropic
        self.has_metallic    = has_metallic
        self.has_spec_tint   = has_spec_tint
        self.specular        = specular
        self.spec_srate = 1.0
        self.diff_refl_srate = 1.0

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, sample1: Float, sample2: mi.Point2f, active: Bool = True) -> tuple[mi.BSDFSample3f, mi.Color3f]:
        return bsdf_sample(si, sample1, sample2, active, self.specular, self.has_anisotropic, self.has_metallic, self.has_spec_tint)
    
    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: Bool = True) -> mi.Color3f:
        return bsdf_eval(si, wo, active, self.specular, self.has_anisotropic, self.has_metallic, self.has_spec_tint)

    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: Bool = True) -> Float:
        return bsdf_pdf(si, wo, active, self.specular, self.has_anisotropic, self.has_metallic, self.spec_srate, self.diff_refl_srate)

    def initialize_mesh_attributes(
            self,
            mesh: mi.Mesh, 
            m_base_color: mi.Color3f | list, 
            m_roughness: float,
            m_metallic: float = None, 
            m_anisotropic: float = None,
            m_spec_tint: float = None) -> list[str]:
        '''
        On a given input triangle mesh, initialize vertex attribute buffers for each of the optimizable 
        BSDF parameters.
        Input:
            - mesh: mi.Mesh. The mesh whose material properties are to be optimized.
            - m_base_color: list[float], size (3,). The initialization value for the mesh's base color.
            - m_roughness, m_metallic, 
              m_anisotropic, m_spec_tint: float. The initialization values for the other scalar parameters
              in the principled BSDF.

        Returns:
            - param_keys: list[str]. List of mesh attribute names corresponding to the BSDF parameters.
        '''
        Nv = mesh.vertex_count()
        color = mi.Color3f([float(x) for x in m_base_color])
        vertex_colors = dr.gather(mi.Color3f, color, dr.zeros(UInt, Nv))

        param_keys = []

        if not(mesh.has_attribute("vertex_bsdf_base_color")):
            mesh.add_attribute("vertex_bsdf_base_color", 3, dr.ravel(vertex_colors))
        param_keys.append("vertex_bsdf_base_color")
        
        if not(mesh.has_attribute("vertex_bsdf_roughness")):
            mesh.add_attribute("vertex_bsdf_roughness", 1, dr.full(Float, m_roughness, Nv))
        param_keys.append("vertex_bsdf_roughness")
        
        if self.has_metallic:
            assert m_metallic is not None, "`m_metallic` is not set!"
            if not(mesh.has_attribute("vertex_bsdf_metallic")):
                mesh.add_attribute("vertex_bsdf_metallic", 1, dr.full(Float, m_metallic, Nv))
            param_keys.append("vertex_bsdf_metallic")

        if self.has_anisotropic:
            assert m_anisotropic is not None, "`m_anisotropic` is not set!"
            if not(mesh.has_attribute("vertex_bsdf_anisotropic")):
                mesh.add_attribute("vertex_bsdf_anisotropic", 1, dr.full(Float, m_anisotropic, Nv))
            param_keys.append("vertex_bsdf_anisotropic")

        if self.has_spec_tint:
            assert m_spec_tint is not None, "`m_spec_tint` is not set!"
            if not(mesh.has_attribute("vertex_bsdf_spec_tint")):
                mesh.add_attribute("vertex_bsdf_spec_tint", 1, dr.full(Float, m_spec_tint, Nv))
            param_keys.append("vertex_bsdf_spec_tint")

        return param_keys


class Diffuse:
    def __init__(self):
        pass

    def sample(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, sample1: Float, sample2: mi.Point2f, active: Bool = True) -> tuple[mi.BSDFSample3f, mi.Color3f]:
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        bs = dr.zeros(mi.BSDFSample3f)

        active &= cos_theta_i > 0.0
        if dr.none(active):
            return bs, 0.0

        bs.wo  = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = +mi.BSDFFlags.DiffuseReflection
        bs.sampled_component = 0

        active &= si.shape.is_mesh()
        value = eval_base_color(si, active)

        return bs, dr.select(active & (bs.pdf > 0.0), value, 0.0)
    
    def eval(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: Bool = True) -> mi.Color3f:
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        active &= (cos_theta_i > 0.0) & (cos_theta_o > 0.0)

        active &= si.shape.is_mesh()
        value = eval_base_color(si, active) * dr.inv_pi * cos_theta_o

        return dr.select(active, value, 0.0)

    def pdf(self, ctx: mi.BSDFContext, si: mi.SurfaceInteraction3f, wo: mi.Vector3f, active: Bool = True) -> Float:
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)

        return dr.select((cos_theta_i > 0.0) & (cos_theta_o > 0.0), pdf, 0.0)

    def initialize_mesh_attributes(
            self,
            mesh: mi.Mesh, 
            m_base_color: mi.Color3f | list) -> list[str]:
        '''
        On a given input triangle mesh, initialize vertex attribute buffers for each of the optimizable 
        BSDF parameters.
        Input:
            - mesh: mi.Mesh. The mesh whose material properties are to be optimized.
            - m_base_color: list[float], size (3,). The initialization value for the mesh's base color.

        Returns:
            - param_keys: list[str]. List of mesh attribute names corresponding to the BSDF parameters.
        '''
        Nv = mesh.vertex_count()
        color = mi.Color3f([float(x) for x in m_base_color])
        vertex_colors = dr.gather(mi.Color3f, color, dr.zeros(UInt, Nv))

        param_keys = []

        if not(mesh.has_attribute("vertex_bsdf_base_color")):
            mesh.add_attribute("vertex_bsdf_base_color", 3, dr.ravel(vertex_colors))
        param_keys.append("vertex_bsdf_base_color")
        return param_keys



import polyscope as ps
from visualizer import plot_mesh_attributes

def visualize_textures(scene: mi.Scene, init_ps = True):
    meshes = [shape for shape in scene.shapes() if shape.is_mesh()]

    if init_ps:
        ps.init()
        # ps.reset_camera_to_home_view()
        ps.look_at([0,0,-4], [0,0,0])

    bsdf_keys = [
        'vertex_bsdf_base_color',
        'vertex_bsdf_roughness',
        'vertex_bsdf_metallic',
        'vertex_bsdf_anisotropic',
        'vertex_bsdf_spec_tint',
        ]
    for idx, mesh in enumerate(meshes):
        keys = [key for key in bsdf_keys if mesh.has_attribute(key)]
        plot_mesh_attributes(mesh, f"mesh{idx}", keys, is_color=True)

    if init_ps:
        ps.show()