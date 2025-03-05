import mitsuba as mi
import drjit as dr

def grayscale_to_color(image: mi.TensorXf):
    shape = dr.shape(image)
    if len(shape) > 3:
        return image
    
    if len(shape) == 3 and shape[2] > 1:
        return image
    
    if len(shape) == 3:
        r,c,_ = shape
    else:
        r,c = shape

    image_color = dr.repeat(image, 3)
    image_color = dr.reshape(mi.TensorXf, image_color, (r, c, 3))
    return image_color

# ### Debug sample ###
# import time
# import numpy as np

# def pretty_print_log(logs):
#     # logs = [log for log in logs if log.get('execution_time', 0) >= 0.05]
#     # Iterate through the logs and format each log entry
#     for entry in logs:
#         print("-" * 80)
#         for key, value in entry.items():
#             # Check for special cases (like StringIO objects)
#             if isinstance(value, str) and value.startswith('<_io.StringIO'):
#                 value = "StringIO object"
#             print(f"{key:20}: {value}")
#         print("-" * 80)

# for it in range(200):
#     time1 = time.time()

#     dr.set_log_level(dr.LogLevel.Info)

#     dr.kernel_history()
#     with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):

#         # Evaluate the objective function for the current BSDF params
#         loss = compute_loss(...)

#         # Backpropagate through the rendering process
#         dr.backward(loss)

#     pretty_print_log(dr.kernel_history())

#     dr.set_log_level(dr.LogLevel.Warn)

#     dr.eval();dr.sync_thread(); time_tmp = time.time()

#     # Optimizer: take a gradient descent step
#     opt.step()

#     # Update the scene state to the new optimized values
#     params.update(opt)

#     time2 = time.time()
#     print(f"Iteration {1+it:03d} [{time2 - time1:2f}]: Loss = {loss}")
