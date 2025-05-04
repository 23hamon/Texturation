#include "utils_cpp.h"

std::vector<float> trace_refract_ray(
    const float* pixel_coord,                   // [x, y]
    const float* air_K_inv_flat,                // 3x3 = 9 floats, row-major
    const float distance,
    const float* normal,                        // [x, y, z]
    const float thickness,
    const float eta_glass,
    const float eta_water
) {
    // Convert inputs to Eigen types
    Eigen::Vector2f pixel_coord_vec(pixel_coord[0], pixel_coord[1]);

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> air_K_inv;
    for (int i = 0; i < 9; ++i){
        air_K_inv(i / 3, i % 3) = air_K_inv_flat[i];
    }
    Eigen::Vector3f normal_vec(normal[0], normal[1], normal[2]);

    // Projected pixel direction in camera
    Eigen::Vector3f pixel_coord_proj(pixel_coord_vec(0), pixel_coord_vec(1), 1.0f);
    Eigen::Vector3f camera_rd = air_K_inv * pixel_coord_proj;
    camera_rd.normalize();

    // First refraction: air glass
    float c = normal_vec.dot(camera_rd);
    float r = 1.0f / eta_glass;
    Eigen::Vector3f glass_rd = r * camera_rd - (r * c - std::sqrt(1.0f - r * r * (1.0f - c * c))) * normal_vec;
    glass_rd.normalize();

    Eigen::Vector3f cam_origin(0.0f, 0.0f, distance);
    Eigen::Vector3f glass_ro = camera_rd * ((cam_origin).dot(normal_vec) / camera_rd.dot(normal_vec));

    // Second refraction: glass water
    c = normal_vec.dot(glass_rd);
    r = eta_glass / eta_water;
    Eigen::Vector3f water_rd = r * glass_rd - (r * c - std::sqrt(1.0f - r * r * (1.0f - c * c))) * normal_vec;
    water_rd.normalize();

    Eigen::Vector3f surface_offset = cam_origin + thickness * normal_vec;
    Eigen::Vector3f water_ro = glass_ro + glass_rd * ((surface_offset - glass_ro).dot(normal_vec) / glass_rd.dot(normal_vec));

    std::vector<float> result(6);
    for (int i = 0; i < 3; ++i) {
        result[i] = water_ro[i];
        result[i + 3] = water_rd[i];
    }
    return result;
}