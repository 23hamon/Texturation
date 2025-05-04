# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

cdef extern from "utils_cpp.h" :
    vector[float] trace_refract_ray(
        const float* pixel_coord,
        const float* air_K_inv,
        const float distance,
        const float* normal,
        const float thickness,
        const float eta_glass,
        const float eta_water,
    )

def py_trace_refract_ray(const float pixel_coord_0,
                         const float pixel_coord_1,
                         air_K_inv, distance,
                         normal, thickness, eta_glass, eta_water):

    cdef float* c_pixel_coord = <float*> malloc(2 * sizeof(float))
    cdef float* c_air_K_inv = <float*> malloc(9 * sizeof(float))
    cdef float* c_normal = <float*> malloc(3 * sizeof(float))
    
    c_pixel_coord[0] = pixel_coord_0
    c_pixel_coord[1] = pixel_coord_1
    for i in range(9):
        c_air_K_inv[i] = air_K_inv[i // 3][i % 3]
    for i in range(3):
        c_normal[i] = normal[i]

    cdef vector[float] r0_rd = trace_refract_ray(c_pixel_coord, c_air_K_inv, distance, c_normal, thickness, eta_glass, eta_water)

    water_ro = [r0_rd[i] for i in range(3)]
    water_rd = [r0_rd[i] for i in range(3, 6)]

    free(c_pixel_coord)
    free(c_air_K_inv)
    free(c_normal)

    return water_ro, water_rd
