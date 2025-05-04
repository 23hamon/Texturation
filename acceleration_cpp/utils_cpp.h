#ifndef UTILS_CPP_H
#define UTILS_CPP_H 

#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include <vector>

std::vector<float> trace_refract_ray(
    const float* pixel_coord,                   // [x, y]
    const float* air_K_inv_flat,                // 3x3 = 9 floats, row-major
    const float distance,
    const float* normal,                        // [x, y, z]
    const float thickness,
    const float eta_glass,
    const float eta_water
) ;


#endif // UTILS_CPP_H