import utils_cpp

ro, rd = utils_cpp.py_trace_refract_ray(
    100.0, 50.0,
    [[1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]],
    0.1,
    [0.0, 0.0, 1.0],
    0.02,
    1.5,
    1.33
)

print("Ray origin:", ro)
print("Ray direction:", rd)
