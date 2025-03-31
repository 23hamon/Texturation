#include <Eigen/Dense>
#include <vector>



//def _stereo_refract_loss_global2(d_l, d_r, Rc, Tc, focals_l, focals_r, air_K_l, air_K_r, n_l, n_r, corners_l, corners_r, thickness, eta_glass, eta_water, objpoints):
float refract_loss_global_impl(
    const float d_l, const float d_r,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rc,
    const Eigen::Vector3f Tc,
    //const Eigen::Array<float, Eigen::Dynamic, 1> focals_l,
    //const Eigen::Array<float, Eigen::Dynamic, 1> focals_r,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> air_K_l,
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> air_K_r,
    const Eigen::Vector3f n_l,
    const Eigen::Vector3f n_r,
    const std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_l,
    const std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_r,
    const float thickness, const float eta_glass, const float eta_water,
    const std::vector<Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>> objpoints
);


float air_loss_global_impl(
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rc, const Eigen::Vector3f Tc,
    /*const Array<float, Dynamic, 1> focals_l, const Array<float, Dynamic, 1> focals_r,*/
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> air_K_l, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> air_K_r,
    const std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_l,
    const std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_r,
    const std::vector<Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>> objpoints
);
