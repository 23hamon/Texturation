#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "global_refract_loss.h"
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;


float refract_loss_global(
    py::array_t<float> x,
    float d_l,
    float d_r,
    py::array_t<float> py_air_K_l,
    py::array_t<float> py_air_K_r,
    py::array_t<float> py_corners_l,
    py::array_t<float> py_corners_r,
    float thickness,
    float eta_glass,
    float eta_water,
    py::array_t<float> py_objpoints
) {
    // TODO abstract the unpacking away to a function
    auto x_acc = x.unchecked<1>();
    auto py_corners_l_acc = py_corners_l.unchecked<3>();
    auto py_corners_r_acc = py_corners_r.unchecked<3>();
    auto py_objpoints_acc = py_objpoints.unchecked<3>();

    Eigen::Vector3f Rc_vec{{x_acc(0), x_acc(1), x_acc(2)}};
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rc = Eigen::AngleAxisf(Rc_vec.norm(), Rc_vec.normalized()).toRotationMatrix();

    Eigen::Vector3f Tc{{x_acc(3), x_acc(4), x_acc(5)}};

    Eigen::Vector3f n_l{{x_acc(6), x_acc(7), x_acc(8)}};
    Eigen::Vector3f n_r{{x_acc(9), x_acc(10), x_acc(11)}};

    n_l.normalize();
    n_r.normalize();

    const unsigned num_images = py_corners_l.shape()[0];
    const unsigned num_corners_per_image = py_corners_l.shape(1);

    Eigen::Vector<float, Eigen::Dynamic> focals_l(num_images), focals_r(num_images);

    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> air_K_l(py_air_K_l.unchecked<2>().data(0, 0));
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> air_K_r(py_air_K_r.unchecked<2>().data(0, 0));

    std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_l;
    std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_r;
    std::vector<Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>> objpoints;

    for (unsigned i = 0; i < num_images; i++) {
        corners_l.emplace_back(Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>>(py_corners_l_acc.data(i, 0, 0), num_corners_per_image, 2));
        corners_r.emplace_back(Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>>(py_corners_r_acc.data(i, 0, 0), num_corners_per_image, 2));
        objpoints.emplace_back(Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>>(py_objpoints_acc.data(i, 0, 0), num_corners_per_image, 3));
    }

    //for (unsigned int i = 0; i < num_images; i++) {
    //    focals_l(i) = x_acc(12 + i * 2);
    //    focals_r(i) = x_acc(13 + i * 2);
    //}

    return refract_loss_global_impl(
        d_l, d_r,
        Rc, Tc,
        //focals_l, focals_r,
        air_K_l, air_K_r,
        n_l, n_r,
        corners_l,
        corners_r,
        thickness, eta_glass, eta_water,
        objpoints
    );
}


float air_loss_global(
    py::array_t<float> x,
    py::array_t<float> py_corners_l,
    py::array_t<float> py_corners_r,
    py::array_t<float> py_objpoints
) {
    const unsigned num_images = py_corners_l.shape()[0];
    const unsigned num_corners_per_image = py_corners_l.shape(1);

    auto x_acc = x.unchecked<1>();
    auto py_corners_l_acc = py_corners_l.unchecked<3>();
    auto py_corners_r_acc = py_corners_r.unchecked<3>();
    auto py_objpoints_acc = py_objpoints.unchecked<3>();

    Eigen::Vector3f Rc_vec{{x_acc(0), x_acc(1), x_acc(2)}};
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rc = Eigen::AngleAxisf(Rc_vec.norm(), Rc_vec.normalized()).toRotationMatrix();

    Eigen::Vector3f Tc{{x_acc(3), x_acc(4), x_acc(5)}};

    // we might actually only wanna start with a single K and try
    // adjusting the focals with penalties as discussed in
    // `global_refract_loss.cpp`
    const float M = 268.73775590551185 / 2.0;
    const float f_l = x_acc(6);
    const float cx_l = x_acc(7);
    const float cy_l = x_acc(8);
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_l {
        {f_l * M, 0.f, cx_l},
        {0.f, f_l * M, cy_l},
        {0.f, 0.f, 1.f}
    };

    const float f_r = x_acc(9);
    const float cx_r = x_acc(10);
    const float cy_r = x_acc(11);
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_r {
        {f_r * M, 0.f, cx_r},
        {0.f, f_r * M, cy_r},
        {0.f, 0.f, 1.f}
    };


    std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_l;
    std::vector<Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>> corners_r;
    std::vector<Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>> objpoints;

    for (unsigned i = 0; i < num_images; i++) {
        corners_l.emplace_back(Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>>(py_corners_l_acc.data(i, 0, 0), num_corners_per_image, 2));
        corners_r.emplace_back(Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 2, Eigen::RowMajor>>(py_corners_r_acc.data(i, 0, 0), num_corners_per_image, 2));
        objpoints.emplace_back(Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>>(py_objpoints_acc.data(i, 0, 0), num_corners_per_image, 3));
    }

    return air_loss_global_impl(
        Rc, Tc,
        K_l,  K_r,
        corners_l,
        corners_r,
        objpoints
    );
}


py::array_t<float> air_loss_global_vectorized(
    py::array_t<float> x,
    py::array_t<float> py_corners_l,
    py::array_t<float> py_corners_r,
    py::array_t<float> py_objpoints
) {
    const py::ssize_t popsize = x.shape(1);
    const py::ssize_t ndim = x.shape(0);
    py::array_t<float> losses{popsize};

    for (py::ssize_t i = 0; i < popsize; i++) {
        py::array_t<float> x_single{ndim};
        for (py::ssize_t j = 0; j < ndim; j++) {
            x_single.mutable_at(j) = x.at(j, i);
        }

        losses.mutable_at(i) = air_loss_global(x_single, py_corners_l, py_corners_r, py_objpoints);
    }

    //exit(0);

    return losses;
}


py::array_t<float> refract_loss_global_vectorized(
    py::array_t<float> x,
    float d_l,
    float d_r,
    py::array_t<float> py_air_K_l,
    py::array_t<float> py_air_K_r,
    py::array_t<float> py_corners_l,
    py::array_t<float> py_corners_r,
    float thickness,
    float eta_glass,
    float eta_water,
    py::array_t<float> py_objpoints
) {
    const py::ssize_t popsize = x.shape(1);
    const py::ssize_t ndim = x.shape(0);
    py::array_t<float> losses{popsize};

    for (py::ssize_t i = 0; i < popsize; i++) {
        py::array_t<float> x_single{ndim};
        for (py::ssize_t j = 0; j < ndim; j++) {
            x_single.mutable_at(j) = x.at(j, i);
        }

        losses.mutable_at(i) = refract_loss_global(
            x_single, d_l, d_r,
            py_air_K_l, py_air_K_r,
            py_corners_l, py_corners_r,
            thickness, eta_glass, eta_water,
            py_objpoints
        );
    }

    return losses;
}


PYBIND11_MODULE(global_refract_loss, m) {
    m.doc() = "bindings for the global refraction loss function";

    m.def("refract_loss_global", &refract_loss_global);
    m.def("air_loss_global", &air_loss_global);
    m.def("air_loss_global_vectorized", &air_loss_global_vectorized);
    m.def("refract_loss_global_vectorized", &refract_loss_global_vectorized);
}


