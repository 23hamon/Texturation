#include "global_refract_loss.h"
#include <utility>
#include <cmath>

// TODO remove
#include <iostream>

using namespace Eigen;


static inline std::pair<Vector3f, Vector3f> trace_refract_ray(
    const Vector2f pixel_coord,
    const Matrix<float, 3, 3, Eigen::RowMajor> air_K_inv,
    const float distance,
    const Vector3f normal,
    const float thickness,
    const float eta_glass,
    const float eta_water
) {
    Vector3f pixel_coord_proj = {pixel_coord(0), pixel_coord(1), 1.};
    Vector3f camera_rd = air_K_inv * pixel_coord_proj;  // TODO performing the same inverse for every point isn't the fastest thing in the world; we could ask the caller to invert it
    camera_rd /= camera_rd.norm();

    // first intersection and refract, from inside the tube (air) to inside the flat port
    float c = normal.dot(camera_rd);
    float r = 1. / eta_glass;
    Vector3f glass_rd = r * camera_rd - (r * c - sqrt(1 - r*r * (1 - c*c))) * normal;
    glass_rd /= glass_rd.norm();
    Vector3f glass_ro = camera_rd * (Vector3f() << 0., 0., distance).finished().dot(normal) / camera_rd.dot(normal);

    // second intersection and refraction, from inside the flat port towards the water
    c = normal.dot(glass_rd);
    r = eta_glass / eta_water;
    Vector3f water_rd = r * glass_rd - (r * c - sqrt(1 - r*r * (1 - c*c))) * normal;
    water_rd /= water_rd.norm();
    Vector3f water_ro = glass_ro + glass_rd * ((Vector3f() << 0., 0., distance).finished() + thickness * normal - glass_ro).dot(normal) / (glass_rd.dot(normal));

    return {water_ro, water_rd};
}


static inline Vector3f trace_air_ray(
    const Vector2f pixel_coord,
    const Matrix<float, 3, 3, Eigen::RowMajor> K_inv
) {
    Vector3f pixel_coord_proj = {pixel_coord(0), pixel_coord(1), 1.};
    Vector3f camera_rd = K_inv * pixel_coord_proj;
    camera_rd /= camera_rd.norm();

    return camera_rd;
}


float refract_loss_global_impl(
    const float d_l, const float d_r,
    const Matrix<float, 3, 3, Eigen::RowMajor> Rc, const Vector3f Tc,
    //const Array<float, Dynamic, 1> focals_l, const Array<float, Dynamic, 1> focals_r,
    const Matrix<float, 3, 3, Eigen::RowMajor> air_K_l, const Matrix<float, 3, 3, Eigen::RowMajor> air_K_r,
    const Vector3f n_l, const Vector3f n_r,
    const std::vector<Array<float, Dynamic, 2, Eigen::RowMajor>> corners_l,
    const std::vector<Array<float, Dynamic, 2, Eigen::RowMajor>> corners_r,
    const float thickness, const float eta_glass, const float eta_water,
    const std::vector<Array<float, Dynamic, 3, Eigen::RowMajor>> objpoints
) {
    const unsigned num_images = corners_l.size();
    const unsigned num_corners_per_image = corners_l[0].rows();  // TODO will segfault if given no corners
    Array<float, Dynamic, 3> losses{num_images * 2 * num_corners_per_image, 3};

    for (unsigned i = 0; i < num_images; i++) {
        Array<float, Dynamic, 3> reconstructed_points1{num_corners_per_image, 3};
        Array<float, Dynamic, 3> reconstructed_points2{num_corners_per_image, 3};

        Vector3f points_mean1 = Vector3f::Zero();
        Vector3f points_mean2 = Vector3f::Zero();

        for (unsigned j = 0; j < num_corners_per_image; j++) {
            Matrix<float, 3, 3, Eigen::RowMajor> modified_air_K_l = air_K_l;
            Matrix<float, 3, 3, Eigen::RowMajor> modified_air_K_r = air_K_r;

            //modified_air_K_l.block(0,0,2,2) *= focals_l(i);
            //modified_air_K_r.block(0,0,2,2) *= focals_r(i);

            Matrix<float, 3, 3, Eigen::RowMajor> modified_air_K_inv_l = modified_air_K_l.inverse();
            Matrix<float, 3, 3, Eigen::RowMajor> modified_air_K_inv_r = modified_air_K_r.inverse();


            auto [ro1, rd1] = trace_refract_ray(corners_l[i].row(j), modified_air_K_inv_l, d_l, n_l, thickness, eta_glass, eta_water);
            auto [ro2, rd2] = trace_refract_ray(corners_r[i].row(j), modified_air_K_inv_r, d_r, n_r, thickness, eta_glass, eta_water);

            ro2 = Rc.transpose() * ro2 - Rc.transpose() * Tc;
            rd2 = Rc.transpose() * rd2;

            // TODO abstract this away to a `closest_point_to_two_lines` function
            Vector3f b = ro2 - ro1;

            // TODO can probably use an eigen cross product here, there should be no performance loss
            float d1_cross_d2_x = rd1(1) * rd2(2) - rd1(2) * rd2(1);
            float d1_cross_d2_y = rd1(2) * rd2(0) - rd1(0) * rd2(2);
            float d1_cross_d2_z = rd1(0) * rd2(1) - rd1(1) * rd2(0);

            float cross_norm2 = d1_cross_d2_x * d1_cross_d2_x + d1_cross_d2_y * d1_cross_d2_y + d1_cross_d2_z * d1_cross_d2_z;
            cross_norm2 = std::max(0.0000001f, cross_norm2);

            float t1 = (b(0) * rd2(1) * d1_cross_d2_z + b(1) * rd2(2) * d1_cross_d2_x + rd2(0) * d1_cross_d2_y * b(2)
                        - b(2) * rd2(1) * d1_cross_d2_x - b(0) * d1_cross_d2_y * rd2(2) - b(1) * rd2(0) * d1_cross_d2_z) / cross_norm2;

            float t2 = (b(0) * rd1(1) * d1_cross_d2_z + b(1) * rd1(2) * d1_cross_d2_x + rd1(0) * d1_cross_d2_y * b(2)
                        - b(2) * rd1(1) * d1_cross_d2_x - b(0) * d1_cross_d2_y * rd1(2) - b(1) * rd1(0) * d1_cross_d2_z) / cross_norm2;

            Vector3f p1 = ro1 + t1 * rd1;
            Vector3f p2 = ro2 + t2 * rd2;

            reconstructed_points1.row(j) = p1;
            reconstructed_points2.row(j) = p2;

            points_mean1 += p1;
            points_mean2 += p2;
        }

        points_mean1 /= num_corners_per_image;
        points_mean2 /= num_corners_per_image;

        Array<float, Dynamic, 3> reconstructed_points = (reconstructed_points1 + reconstructed_points2) / 2.;
        Vector3f points_mean = (points_mean1 + points_mean2) / 2.;

        Array<float, Dynamic, 3> reconstructed_points_centered = reconstructed_points.matrix().rowwise() - points_mean.transpose();
        Array<float, Dynamic, 3> img_objpoints_centered = objpoints[i];

        Matrix<float, 3, 3, Eigen::RowMajor> H = reconstructed_points_centered.matrix().transpose() * img_objpoints_centered.matrix();
        JacobiSVD<Matrix<float, 3, 3, Eigen::RowMajor>> svd(H, ComputeFullU | ComputeFullV);
        Matrix<float, 3, 3, Eigen::RowMajor> U = svd.matrixU();
        Matrix<float, 3, 3, Eigen::RowMajor> V = svd.matrixV();

        U /= abs(U.determinant());
        V /= abs(V.determinant());

        if (U.determinant() * V.determinant() < 0.) {
            U(all, last) *= -1.;
        }

        Matrix<float, 3, 3, Eigen::RowMajor> rot = U * V;

        for (unsigned j = 0; j < num_corners_per_image; j++) {
            const Vector3f rotated_point = rot.transpose() * reconstructed_points_centered.row(j).matrix().transpose();
            losses.row(i * 2 * num_corners_per_image + 2 * j) = rotated_point - img_objpoints_centered.row(j).matrix().transpose();
            losses.row(i * 2 * num_corners_per_image + 2 * j + 1) = reconstructed_points2.row(j) - reconstructed_points1.row(j);
        }
    }


    losses *= losses;

    //return (losses + 1.).log().sum();
    return (losses * losses).sum();
}


float air_loss_global_impl(
    const Matrix<float, 3, 3, Eigen::RowMajor> Rc, const Vector3f Tc,
    /*const Array<float, Dynamic, 1> focals_l, const Array<float, Dynamic, 1> focals_r,*/
    const Matrix<float, 3, 3, Eigen::RowMajor> air_K_l, const Matrix<float, 3, 3, Eigen::RowMajor> air_K_r,
    const std::vector<Array<float, Dynamic, 2, Eigen::RowMajor>> corners_l,
    const std::vector<Array<float, Dynamic, 2, Eigen::RowMajor>> corners_r,
    const std::vector<Array<float, Dynamic, 3, Eigen::RowMajor>> objpoints
) {
    const unsigned num_images = corners_l.size();
    const unsigned num_corners_per_image = corners_l[0].rows();  // TODO will segfault if given no corners
    Array<float, Dynamic, 3> losses{num_images * 2 * num_corners_per_image, 3};

    const Matrix<float, 3, 3, Eigen::RowMajor> air_K_l_inv = air_K_l.inverse();
    const Matrix<float, 3, 3, Eigen::RowMajor> air_K_r_inv = air_K_r.inverse();

    for (unsigned i = 0; i < num_images; i++) {
        Array<float, Dynamic, 3> reconstructed_points1{num_corners_per_image, 3};
        Array<float, Dynamic, 3> reconstructed_points2{num_corners_per_image, 3};

        Vector3f points_mean1 = Vector3f::Zero();
        Vector3f points_mean2 = Vector3f::Zero();

        for (unsigned j = 0; j < num_corners_per_image; j++) {
            Vector3f rd1 = trace_air_ray(corners_l[i].row(j), air_K_l_inv);
            Vector3f rd2 = trace_air_ray(corners_r[i].row(j), air_K_r_inv);

            Vector3f ro1 = {0.f, 0.f, 0.f};
            Vector3f ro2 = - Rc.transpose() * Tc;
            rd2 = Rc.transpose() * rd2;

            // TODO abstract this away to a `closest_point_to_two_lines` function
            Vector3f b = ro2 - ro1;

            // TODO can probably use an eigen cross product here, there should be no performance loss
            float d1_cross_d2_x = rd1(1) * rd2(2) - rd1(2) * rd2(1);
            float d1_cross_d2_y = rd1(2) * rd2(0) - rd1(0) * rd2(2);
            float d1_cross_d2_z = rd1(0) * rd2(1) - rd1(1) * rd2(0);

            float cross_norm2 = d1_cross_d2_x * d1_cross_d2_x + d1_cross_d2_y * d1_cross_d2_y + d1_cross_d2_z * d1_cross_d2_z;
            cross_norm2 = std::max(0.0000001f, cross_norm2);

            float t1 = (b(0) * rd2(1) * d1_cross_d2_z + b(1) * rd2(2) * d1_cross_d2_x + rd2(0) * d1_cross_d2_y * b(2)
                        - b(2) * rd2(1) * d1_cross_d2_x - b(0) * d1_cross_d2_y * rd2(2) - b(1) * rd2(0) * d1_cross_d2_z) / cross_norm2;

            float t2 = (b(0) * rd1(1) * d1_cross_d2_z + b(1) * rd1(2) * d1_cross_d2_x + rd1(0) * d1_cross_d2_y * b(2)
                        - b(2) * rd1(1) * d1_cross_d2_x - b(0) * d1_cross_d2_y * rd1(2) - b(1) * rd1(0) * d1_cross_d2_z) / cross_norm2;

            Vector3f p1 = ro1 + t1 * rd1;
            Vector3f p2 = ro2 + t2 * rd2;

            reconstructed_points1.row(j) = p1;
            reconstructed_points2.row(j) = p2;

            points_mean1 += p1;
            points_mean2 += p2;
        }

        points_mean1 /= num_corners_per_image;
        points_mean2 /= num_corners_per_image;

        Array<float, Dynamic, 3> reconstructed_points = (reconstructed_points1 + reconstructed_points2) / 2.;
        Vector3f points_mean = (points_mean1 + points_mean2) / 2.;

        Array<float, Dynamic, 3> reconstructed_points_centered = reconstructed_points.matrix().rowwise() - points_mean.transpose();
        Array<float, Dynamic, 3> img_objpoints_centered = objpoints[i];

        Matrix<float, 3, 3, Eigen::RowMajor> H = reconstructed_points_centered.matrix().transpose() * img_objpoints_centered.matrix();
        JacobiSVD<Matrix<float, 3, 3, Eigen::RowMajor>> svd(H, ComputeFullU | ComputeFullV);
        Matrix<float, 3, 3, Eigen::RowMajor> U = svd.matrixU();
        Matrix<float, 3, 3, Eigen::RowMajor> V = svd.matrixV();
        if (U.determinant() * V.determinant() < 0.) {
            U(all, last) *= -1.;
        }

        Matrix<float, 3, 3, Eigen::RowMajor> rot = U * V;

        for (unsigned j = 0; j < num_corners_per_image; j++) {
            const Vector3f rotated_point = rot.transpose() * reconstructed_points_centered.row(j).matrix().transpose();
            losses.row(i * 2 * num_corners_per_image + 2 * j) = rotated_point - img_objpoints_centered.row(j).matrix().transpose();
            losses.row(i * 2 * num_corners_per_image + 2 * j + 1) = reconstructed_points2.row(j) - reconstructed_points1.row(j);
        }
    }


    losses *= losses;

    //return (losses + 1.).log().sum();
    return (losses * losses).sum();
}
