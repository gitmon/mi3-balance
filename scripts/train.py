import mitsuba as mi
import drjit as dr

def define_BSDF_diffuse_textured(tex_size: tuple[int] = (64, 64), tex = None):
    blank_grid = mi.Bitmap(dr.full(mi.TensorXf, 0.5, shape=(*tex_size, 3)))
    bsdf = mi.load_dict({
        "type": "diffuse",
        "reflectance": {
            "type": "bitmap",
            "id": "albedo_tex",
            "bitmap": tex if tex is not None else blank_grid,
            # IMPORTANT: for low-res textures (e.g. 8x8), use interpolation mode "nearest"
            # "filter_type": "bilinear",   
            "filter_type": "nearest",   
            "raw": True
        }
    })
    param_keys = ["reflectance.data"]
    return bsdf, param_keys

def define_BSDF_diffuse_uniform(color_init = [0.5, 0.5, 0.5]):
    bsdf = mi.load_dict({
        "type": "diffuse",
        "reflectance": {
            "type": "rgb",
            "value": color_init
        }
    })
    param_keys = ["reflectance.value"]
    return bsdf, param_keys

def define_BSDF_principled_uniform(color_init = [0.5, 0.5, 0.5], roughness_init = 0.5):
    bsdf = mi.load_dict({
        "type": "principled",
        "base_color": {
            "type": "rgb",
            "value": color_init
        },
        "roughness": roughness_init,
    })
    param_keys = ["base_color.value", "roughness.value"]
    return bsdf, param_keys