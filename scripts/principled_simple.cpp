#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/texture.h>
#include "principledhelpers.h"

NAMESPACE_BEGIN(mitsuba)
/**!
.. _bsdf-principled:

The Principled BSDF (:monosp:`principled`)
-----------------------------------------------------
.. pluginparameters::

 * - base_color
   - |spectrum| or |texture|
   - The color of the material. (Default:0.5)
   - |exposed|, |differentiable|

 * - roughness
   - |float| or |texture|
   - Controls the roughness parameter of the main specular lobes. (Default:0.5)
   - |exposed|, |differentiable|, |discontinuous|

 * - anisotropic
   - |float| or |texture|
   - Controls the degree of anisotropy. (0.0 : isotropic material) (Default:0.0)
   - |exposed|, |differentiable|, |discontinuous|

 * - metallic
   - |texture| or |float|
   - The "metallicness" of the model. (Default:0.0)
   - |exposed|, |differentiable|, |discontinuous|

 * - spec_trans
   - |texture| or |float|
   - Blends BRDF and BSDF major lobe. (1.0: only BSDF
     response, 0.0 : only BRDF response.) (Default: 0.0)
   - |exposed|, |differentiable|, |discontinuous|

 * - eta
   - |float|
   - Interior IOR/Exterior IOR
   - |exposed|, |differentiable|, |discontinuous|

 * - specular
   - |float|
   - Controls the Fresnel reflection coefficient. This parameter has one to one
     correspondence with `eta`, so both of them can not be specified in xml.
     (Default:0.5)
   - |exposed|, |differentiable|, |discontinuous|

 * - spec_tint
   - |texture| or |float|
   - The fraction of `base_color` tint applied onto the dielectric reflection
     lobe. (Default:0.0)
   - |exposed|, |differentiable|

 * - sheen
   - |float| or |texture|
   - The rate of the sheen lobe. (Default:0.0)
   - |exposed|, |differentiable|

 * - sheen_tint
   - |float| or |texture|
   - The fraction of `base_color` tint applied onto the sheen lobe. (Default:0.0)
   - |exposed|, |differentiable|

 * - flatness
   - |float| or |texture|
   - Blends between the diffuse response and fake subsurface approximation based
     on Hanrahan-Krueger approximation. (0.0:only diffuse response, 1.0:only
     fake subsurface scattering.) (Default:0.0)
   - |exposed|, |differentiable|

 * - clearcoat
   - |texture| or |float|
   - The rate of the secondary isotropic specular lobe. (Default:0.0)
   - |exposed|, |differentiable|, |discontinuous|

 * - clearcoat_gloss
   - |texture| or |float|
   - Controls the roughness of the secondary specular lobe. Clearcoat response
     gets glossier as the parameter increases. (Default:0.0)
   - |exposed|, |differentiable|, |discontinuous|

 * - diffuse_reflectance_sampling_rate
   - |float|
   - The rate of the cosine hemisphere reflection in sampling. (Default:1.0)
   - |exposed|

 * - main_specular_sampling_rate
   - |float|
   - The rate of the main specular lobe in sampling. (Default:1.0)
   - |exposed|

 * - clearcoat_sampling_rate
   - |float|
   - The rate of the secondary specular reflection in sampling. (Default:0.0)
   - |exposed|

The principled BSDF is a complex BSDF with numerous reflective and transmissive
lobes. It is able to produce great number of material types ranging from metals
to rough dielectrics. Moreover, the set of input parameters are designed to be
artist-friendly and do not directly correspond to physical units.

The implementation is based on the papers *Physically Based Shading at Disney*
:cite:`Disney2012` and *Extending the Disney BRDF to a BSDF with Integrated
Subsurface Scattering* :cite:`Disney2015` by Brent Burley.

 .. note::

    Subsurface scattering and volumetric extinction is not supported!

Images below show how the input parameters affect the appearance of the objects
while one of the parameters is changed for each column.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/principled_blend.png
   :caption: Blending of parameters when transmission lobe is turned off.
.. subfigure:: ../../resources/data/docs/images/render/principled_st_blend.png
    :caption: Blending of parameters when transmission lobe is turned on.
.. subfigend::
    :label: fig-blend-principled

You can see the general structure of the BSDF below.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/bsdf/principled.png
    :caption: The general structure of the principled BSDF
.. subfigend::
    :label: fig-structure-principled

The following XML snippet describes a material definition for :monosp:`principled`
material:

.. tabs::
    .. code-tab:: xml
        :name: principled

        <bsdf type="principled">
            <rgb name="base_color" value="1.0,1.0,1.0"/>
            <float name="metallic" value="0.7" />
            <float name="specular" value="0.6" />
            <float name="roughness" value="0.2" />
            <float name="spec_tint" value="0.4" />
            <float name="anisotropic" value="0.5" />
            <float name="sheen" value="0.3" />
            <float name="sheen_tint" value="0.2" />
            <float name="clearcoat" value="0.6" />
            <float name="clearcoat_gloss" value="0.3" />
            <float name="spec_trans" value="0.4" />
        </bsdf>

    .. code-tab:: python

        'type': 'principled',
        'base_color': {
            'type': 'rgb',
            'value': [1.0, 1.0, 1.0]
        },
        'metallic': 0.7,
        'specular': 0.6,
        'roughness': 0.2,
        'spec_tint': 0.4,
        'anisotropic': 0.5,
        'sheen': 0.3,
        'sheen_tint': 0.2,
        'clearcoat': 0.6,
        'clearcoat_gloss': 0.3,
        'spec_trans': 0.4

All of the parameters except sampling rates and `eta` should take values
between 0.0 and 1.0.
 */


/**
 * \brief Calculates the microfacet distribution parameters based on
 * Disney Course Notes.
 * \param anisotropic
 *     Anisotropy weight.
 * \param roughness
 *     Roughness parameter of the material.
 * \return Microfacet Distribution roughness parameters: alpha_x, alpha_y.
 */
template<typename Float>
std::pair<Float, Float> calc_dist_params(Float anisotropic,
                                         Float roughness,
                                         bool has_anisotropic){
    Float roughness_2 = dr::square(roughness);
    if (!has_anisotropic) {
        Float a = dr::maximum(0.001f, roughness_2);
        return { a, a };
    }
    Float aspect = dr::sqrt(1.0f - 0.9f * anisotropic);
    return { dr::maximum(0.001f, roughness_2 / aspect),
             dr::maximum(0.001f, roughness_2 * aspect) };
}


/**
 * \brief Computes a mask for macro-micro surface incompatibilities.
 * \param m
 *     Micro surface normal.
 * \param wi
 *     Incident direction.
 * \param wo
 *     Outgoing direction.
 * \param cos_theta_i
 *     Incident angle
 * \param reflection
 *     Flag for determining reflection or refraction case.
 * \return  Macro-micro surface compatibility mask.
 */
template <typename Float>
dr::mask_t<Float> mac_mic_compatibility(const Vector<Float,3> &m,
                                        const Vector<Float,3> &wi,
                                        const Vector<Float,3> &wo,
                                        const Float &cos_theta_i,
                                        bool reflection) {
    if (reflection) {
        return (dr::dot(wi, dr::mulsign(m, cos_theta_i)) > 0.0f) &&
        (dr::dot(wo, dr::mulsign(m, cos_theta_i)) > 0.0f);
    } else {
        return (dr::dot(wi, dr::mulsign(m, cos_theta_i)) > 0.0f) &&
        (dr::dot(wo, dr::mulsign_neg(m, cos_theta_i)) > 0.0f);
    }
}



/**
 * \brief  Modified fresnel function for the principled material. It blends
 * metallic and dielectric responses (not true metallic). spec_tint portion
 * of the dielectric response is tinted towards base_color. Schlick
 * approximation is used for spec_tint and metallic parts whereas dielectric
 * part is calculated with the true fresnel dielectric implementation.
 * \param F_dielectric
 *     True dielectric response.
 * \param metallic
 *     Metallic weight.
 * \param spec_tint
 *     Specular tint weight.
 * \param base_color
 *     Base color of the material.
 * \param lum
 *     Luminance of the base color.
 * \param cos_theta_i
 *     Incident angle of the ray based on microfacet normal.
 * \param front_side
 *     Mask for front side of the macro surface.
 * \param bsdf
 *     Weight of the BSDF major lobe.
 * \return Fresnel term of principled BSDF with metallic and dielectric response
 * combined.
 */
template<typename Float,typename T>
T principled_fresnel(const Float &F_dielectric, const Float &metallic,
                     const Float &spec_tint,
                     const T &base_color,
                     const Float &lum, const Float &cos_theta_i,
                     const dr::mask_t<Float> &front_side,
                     const Float &bsdf, const Float &eta,
                     bool has_metallic, bool has_spec_tint) {
    // Outside mask based on micro surface
    dr::mask_t<Float> outside_mask = cos_theta_i >= 0.0f;
    Float rcp_eta = dr::rcp(eta);
    Float eta_it  = dr::select(outside_mask, eta, rcp_eta);
    T F_schlick(0.0f);

    // Metallic component based on Schlick.
    if (has_metallic) {
        F_schlick += metallic * calc_schlick<T>(
                base_color, cos_theta_i,eta);
    }

    // Tinted dielectric component based on Schlick.
    if (has_spec_tint) {
        T c_tint       =
                dr::select(lum > 0.0f, base_color / lum, 1.0f);
        T F0_spec_tint =
                c_tint * schlick_R0_eta(eta_it);
        F_schlick +=
                (1.0f - metallic) * spec_tint *
                calc_schlick<T>(F0_spec_tint, cos_theta_i,eta);
    }

    // Front side fresnel.
    T F_front =
            (1.0f - metallic) * (1.0f - spec_tint) * F_dielectric + F_schlick;
    /* For back side there is no tint or metallic, just true dielectric
       fresnel.*/
    return dr::select(front_side, F_front, bsdf * F_dielectric);
}


/**
 * \brief Approximation of incident specular based on relative index of
 * refraction.
 * \param eta
 *     Relative index of refraction.
 * \return Incident specular
 */
template <typename Float>
Float schlick_R0_eta(Float eta){
    return dr::square((eta - 1.0f) / (eta + 1.0f));
}


/**
 * \brief Schlick Approximation for Fresnel Reflection coefficient F = R0 +
 * (1-R0) (1-cos^5(i)). Transmitted ray's angle should be used for eta<1.
 * \param R0
 *     Incident specular. (Fresnel term when incident ray is aligned with
 *     the surface normal.)
 * \param cos_theta_i
 *     Incident angle of the ray based on microfacet normal.
 * \return Schlick approximation result.
 */
template <typename T,typename Float>
T calc_schlick(T R0, Float cos_theta_i,Float eta){
    dr::mask_t<Float> outside_mask = cos_theta_i >= 0.0f;
    Float rcp_eta     = dr::rcp(eta),
    eta_it      = dr::select(outside_mask, eta, rcp_eta),
    eta_ti      = dr::select(outside_mask, rcp_eta, eta);

    Float cos_theta_t_sqr = dr::fnmadd(
            dr::fnmadd(cos_theta_i, cos_theta_i, 1.0f), dr::square(eta_ti), 1.0f);
    Float cos_theta_t = dr::safe_sqrt(cos_theta_t_sqr);
    return dr::select(
            eta_it > 1.0f,
            dr::lerp(schlick_weight(dr::abs(cos_theta_i)), 1.0f, R0),
            dr::lerp(schlick_weight(cos_theta_t), 1.0f, R0));
}


/**
 * \brief Computes the schlick weight for Fresnel Schlick approximation.
 * \param cos_i
 *     Incident angle of the ray based on microfacet normal.
 * \return schlick weight
 */
template <typename Float>
Float schlick_weight(Float cos_i) {
    Float m = dr::clip(1.0f - cos_i, 0.0f, 1.0f);
    return dr::square(dr::square(m)) * m;
}


template <typename Float, typename Spectrum>
class Principled final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)
    using GTR1 = GTR1Isotropic<Float, Spectrum>;

    Principled(const Properties &props) : Base(props) {
        // Parameter definitions
        m_base_color = props.texture<Texture>("base_color", 0.5f);      // keep
        m_roughness = props.texture<Texture>("roughness", 0.5f);        // keep
        m_has_anisotropic = get_flag("anisotropic", props);             // true/false
        m_anisotropic = props.texture<Texture>("anisotropic", 0.0f);    // keep?
        m_has_metallic = get_flag("metallic", props);
        m_metallic = props.texture<Texture>("metallic", 0.0f);

        // Not sure
        m_has_spec_tint = get_flag("spec_tint", props);
        m_spec_tint = props.texture<Texture>("spec_tint", 0.0f);
        m_specular = props.get<float>("specular", 0.5f);
        m_eta = 2.0f * dr::rcp(1.0f - dr::sqrt(0.08f * m_specular)) - 1.0f;

        // Not important...?
        m_spec_srate = 1.0f;
        m_diff_refl_srate = 1.0f;

        initialize_lobes();
    }

    void initialize_lobes() {
        // Diffuse reflection lobe
        m_components.push_back(BSDFFlags::DiffuseReflection |
                               BSDFFlags::FrontSide);

        // Main specular reflection lobe
        uint32_t f = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide |
                     BSDFFlags::BackSide;
        if (m_has_anisotropic)
            f = f | BSDFFlags::Anisotropic;
        m_components.push_back(f);
    }


    std::pair<BSDFSample3f, Spectrum>
    sample(const SurfaceInteraction3f &si,
           Float sample1, const Point2f &sample2, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs   = dr::zeros<BSDFSample3f>();

        // Ignoring perfectly grazing incoming rays
        active &= cos_theta_i != 0.0f;

        if (unlikely(dr::none_or<false>(active)))
            return { bs, 0.0f };

        // Store the weights.
        Float anisotropic = m_has_anisotropic ? m_anisotropic->eval_1(si, active) : 0.0f,
        roughness = m_roughness->eval_1(si, active),
        metallic = m_has_metallic ? m_metallic->eval_1(si, active) : 0.0f;

        // Weights of BSDF and BRDF major lobes
        Float brdf = (1.0f - metallic);

        // Mask for incident side. (wi.z<0)
        Mask front_side = cos_theta_i > 0.0f;

        // Defining main specular reflection distribution
        auto [ax, ay] = calc_dist_params(anisotropic, roughness, m_has_anisotropic);
        MicrofacetDistribution spec_distr(MicrofacetType::GGX, ax, ay);
        Normal3f m_spec = std::get<0>(
                spec_distr.sample(dr::mulsign(si.wi, cos_theta_i), sample2));

        // Fresnel coefficient for the main specular.
        auto [F_spec_dielectric, cos_theta_t, eta_it, eta_ti] =
                fresnel(dr::dot(si.wi, m_spec), m_eta);

        // If BSDF major lobe is turned off, we do not sample the inside
        // case.
        active &= front_side;

        // Probability definitions
        /* Inside  the material, just microfacet Reflection and
           microfacet Transmission is sampled. */
        Float prob_spec_reflect = dr::select(
                front_side,
                m_spec_srate,
                F_spec_dielectric);
        Float prob_diffuse = dr::select(front_side, brdf * m_diff_refl_srate, 0.0f);

        // Normalizing the probabilities.
        prob_diffuse *= dr::rcp(prob_spec_reflect + prob_diffuse);

        // Sampling mask definitions
        Float curr_prob(0.0f);
        Mask sample_diffuse = active && (sample1 < prob_diffuse);
        curr_prob += prob_diffuse;
        Mask sample_spec_reflect = active && (sample1 >= curr_prob);

        // Eta will be changed in transmission.
        bs.eta = 1.0f;

        // Main specular reflection sampling
        if (dr::any_or<true>(sample_spec_reflect)) {
            Vector3f wo                            = reflect(si.wi, m_spec);
            dr::masked(bs.wo, sample_spec_reflect) = wo;
            dr::masked(bs.sampled_component, sample_spec_reflect) = 3;
            dr::masked(bs.sampled_type, sample_spec_reflect) =
                    +BSDFFlags::GlossyReflection;

            /* Filter the cases where macro and micro surfaces do not agree
             on the same side and reflection is not successful*/
            Mask reflect = cos_theta_i * Frame3f::cos_theta(wo) > 0.0f;
            active &=
                    (!sample_spec_reflect ||
                    (mac_mic_compatibility(Vector3f(m_spec),
                                           si.wi, wo, cos_theta_i, true) &&
                    reflect));
        }
        // Cosine hemisphere reflection sampling
        if (dr::any_or<true>(sample_diffuse)) {
            Vector3f wo = warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.wo, sample_diffuse)                = wo;
            dr::masked(bs.sampled_component, sample_diffuse) = 0;
            dr::masked(bs.sampled_type, sample_diffuse) =
                    +BSDFFlags::DiffuseReflection;
            Mask reflect = cos_theta_i * Frame3f::cos_theta(wo) > 0.0f;
            active &= (!sample_diffuse || reflect);
        }

        bs.pdf = pdf(si, bs.wo, active);
        active &= bs.pdf > 0.0f;
        Spectrum result = eval(si, bs.wo, active);
        return { bs, result / bs.pdf & active };
    }

    Spectrum eval(const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        // Ignore perfectly grazing configurations
        active &= cos_theta_i != 0.0f;

        if (unlikely(dr::none_or<false>(active)))
            return 0.0f;

        // Store the weights.
        Float anisotropic = m_has_anisotropic ? m_anisotropic->eval_1(si, active) : 0.0f,
              roughness = m_roughness->eval_1(si, active),
              metallic = m_has_metallic ? m_metallic->eval_1(si, active) : 0.0f;
        UnpolarizedSpectrum base_color = m_base_color->eval(si, active);

        // Weights for BRDF and BSDF major lobes.
        Float brdf = (1.0f - metallic),
        bsdf = 0.0f;

        Float cos_theta_o = Frame3f::cos_theta(wo);

        // Reflection and refraction masks.
        Mask reflect = cos_theta_i * cos_theta_o > 0.0f;
        Mask refract = cos_theta_i * cos_theta_o < 0.0f;

        // Masks for the side of the incident ray (wi.z<0)
        Mask front_side = cos_theta_i > 0.0f;
        Float inv_eta   = dr::rcp(m_eta);

        // Eta value w.r.t. ray instead of the object.
        Float eta_path     = dr::select(front_side, m_eta, inv_eta);
        Float inv_eta_path = dr::select(front_side, inv_eta, m_eta);

        // Main specular reflection and transmission lobe
        auto [ax, ay] = calc_dist_params(anisotropic, roughness, m_has_anisotropic);
        MicrofacetDistribution spec_dist(MicrofacetType::GGX, ax, ay);

        // Halfway vector
        Vector3f wh =
                dr::normalize(si.wi + wo * dr::select(reflect, 1.0f, eta_path));

        // Make sure that the halfway vector points outwards the object
        wh = dr::mulsign(wh, Frame3f::cos_theta(wh));

        // Dielectric Fresnel
        auto [F_spec_dielectric, cos_theta_t, eta_it, eta_ti] =
                fresnel(dr::dot(si.wi, wh), m_eta);

        Mask reflection_compatibilty =
                mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, true);
        Mask refraction_compatibilty =
                mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, false);
        // Masks for evaluating the lobes.
        // Specular reflection mask
        Mask spec_reflect_active = active && reflect &&
                reflection_compatibilty &&
                (F_spec_dielectric > 0.0f);

        // Diffuse, retro and fake subsurface mask
        Mask diffuse_active = active && (brdf > 0.0f) && reflect && front_side;

        // Evaluate the microfacet normal distribution
        Float D = spec_dist.eval(wh);

        // Smith's shadowing-masking function
        Float G = spec_dist.G(si.wi, wo, wh);

        // Initialize the final BSDF value.
        UnpolarizedSpectrum value(0.0f);

        // Main specular reflection evaluation
        if (dr::any_or<true>(spec_reflect_active)) {
            // No need to calculate luminance if there is no color tint.
            Float lum = m_has_spec_tint
                    ? mitsuba::luminance(base_color, si.wavelengths)
                    : 1.0f;
            Float spec_tint =
                    m_has_spec_tint ? m_spec_tint->eval_1(si, active) : 0.0f;

            // Fresnel term
            UnpolarizedSpectrum F_principled = principled_fresnel(
                    F_spec_dielectric, metallic, spec_tint, base_color, lum,
                    dr::dot(si.wi, wh), front_side, bsdf, m_eta, m_has_metallic,
                    m_has_spec_tint);

            // Adding the specular reflection component
            dr::masked(value, spec_reflect_active) +=
                    F_principled * D * G / (4.0f * dr::abs(cos_theta_i));
        }

        // Evaluation of diffuse, retro reflection, fake subsurface and
        // sheen.
        if (dr::any_or<true>(diffuse_active)) {
            Float Fo = schlick_weight(dr::abs(cos_theta_o)),
            Fi = schlick_weight(dr::abs(cos_theta_i));

            // Diffuse
            Float f_diff = (1.0f - 0.5f * Fi) * (1.0f - 0.5f * Fo);

            Float cos_theta_d = dr::dot(wh, wo);
            Float Rr          = 2.0f * roughness * dr::square(cos_theta_d);

            // Retro reflection
            Float f_retro = Rr * (Fo + Fi + Fo * Fi * (Rr - 1.0f));

            // Adding diffuse, retro evaluation. (no fake ss.)
            dr::masked(value, diffuse_active) +=
                    brdf * dr::abs(cos_theta_o) * base_color *
                    dr::InvPi<Float> * (f_diff + f_retro);
        }
        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        // Ignore perfectly grazing configurations.
        active &= cos_theta_i != 0.0f;

        if (unlikely(dr::none_or<false>(active)))
            return 0.0f;

        // Store the weights.
        Float anisotropic =
                m_has_anisotropic ? m_anisotropic->eval_1(si, active) : 0.0f,
                roughness = m_roughness->eval_1(si, active);
        Float metallic = m_has_metallic ? m_metallic->eval_1(si, active) : 0.0f;

        // BRDF and BSDF major lobe weights
        Float brdf = (1.0f - metallic);

        // Masks if incident direction is inside (wi.z<0)
        Mask front_side = cos_theta_i > 0.0f;

        // Eta w.r.t. light path.
        Float eta_path    = dr::select(front_side, m_eta, dr::rcp(m_eta));
        Float cos_theta_o = Frame3f::cos_theta(wo);

        Mask reflect = cos_theta_i * cos_theta_o > 0.0f;
        Mask refract = cos_theta_i * cos_theta_o < 0.0f;

        // Halfway vector calculation
        Vector3f wh = dr::normalize(
                si.wi + wo * dr::select(reflect, Float(1.0f), eta_path));

        // Make sure that the halfway vector points outwards the object
        wh = dr::mulsign(wh, Frame3f::cos_theta(wh));

        // Main specular distribution for reflection and transmission.
        auto [ax, ay] = calc_dist_params(anisotropic, roughness,m_has_anisotropic);
        MicrofacetDistribution spec_distr(MicrofacetType::GGX, ax, ay);

        // Dielectric Fresnel calculation
        auto [F_spec_dielectric, cos_theta_t, eta_it, eta_ti] =
                fresnel(dr::dot(si.wi, wh), m_eta);

        // Defining the probabilities
        Float prob_spec_reflect = dr::select(
                front_side,
                m_spec_srate,
                F_spec_dielectric);
        Float prob_diffuse = dr::select(front_side, brdf * m_diff_refl_srate, 0.f);

        // Normalizing the probabilities.
        Float rcp_tot_prob = dr::rcp(prob_spec_reflect + prob_diffuse);
        prob_spec_reflect *= rcp_tot_prob;
        prob_diffuse *= rcp_tot_prob;

        /* Calculation of dwh/dwo term. Different for reflection and
         transmission. */
        Float dwh_dwo_abs = dr::abs(dr::rcp(4.0f * dr::dot(wo, wh)));

        // Initializing the final pdf value.
        Float pdf(0.0f);

        // Macro-micro surface compatibility mask for reflection.
        Mask mfacet_reflect_macmic =
                mac_mic_compatibility(wh, si.wi, wo, cos_theta_i, true) && reflect;

        // Adding main specular reflection pdf
        dr::masked(pdf, mfacet_reflect_macmic) +=
                prob_spec_reflect *
                spec_distr.pdf(dr::mulsign(si.wi, cos_theta_i), wh) * dwh_dwo_abs;
        // Adding cosine hemisphere reflection pdf
        dr::masked(pdf, reflect) +=
                prob_diffuse * warp::square_to_cosine_hemisphere_pdf(wo);

        return pdf;
    }

    MI_DECLARE_CLASS()
private:
    /// Parameters
    ref<Texture> m_base_color;
    ref<Texture> m_roughness;
    ref<Texture> m_anisotropic;
    ref<Texture> m_spec_tint;
    ref<Texture> m_metallic;
    Float m_eta;
    Float m_specular;

    /// Sampling rates
    ScalarFloat m_diff_refl_srate;
    ScalarFloat m_spec_srate;

    /// Whether the lobes are active or not.
    bool m_has_metallic;
    bool m_has_spec_tint;
    bool m_has_anisotropic;
};

MI_IMPLEMENT_CLASS_VARIANT(Principled, BSDF)
MI_EXPORT_PLUGIN(Principled, "The Principled Material")
NAMESPACE_END(mitsuba)
