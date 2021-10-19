///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, IRI (CSIC - UPC)
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/quadratic-full.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationQuadFull() {
  boost::python::register_ptr_to_python<boost::shared_ptr<ActivationModelQuadFull> >();

  bp::class_<ActivationModelQuadFull, bp::bases<ActivationModelAbstract> >(
      "ActivationModelQuadFull",
      "Inequality activation model.\n\n"
      "The activation has a linear and quadratic term\n"
      "a(r) = w^T * r + 0.5 * r^T*W*r \n"
      "where w a vector of weights and W is a dense matrix of weights",
      bp::init<Eigen::VectorXd, Eigen::MatrixXd>(bp::args("self", "linear_weights", "quadratic_weights"),
                                                 "Initialize the activation model.\n\n"
                                                 ":param linear weights: linear coefficients\n"
                                                 ":param quadratic_weights: quadratic coefficients"))
      .def("calc", &ActivationModelQuadFull::calc, bp::args("self", "data", "r"),
           "Compute the inequality activation.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelQuadFull::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of inequality activation.\n\n"
           ":param data: activation data\n"
           "Note that the Hessian is constant, so we don't write again this value.\n"
           "It assumes that calc has been run first.\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelQuadFull::createData, bp::args("self"),
           "Create the weighted quadratic action data.")
      .add_property("linear_weights",
                    bp::make_function(&ActivationModelQuadFull::get_linear_weights, bp::return_internal_reference<>()),
                    bp::make_function(&ActivationModelQuadFull::set_linear_weights), "vector of weights")
      .add_property(
          "quadratic_weights",
          bp::make_function(&ActivationModelQuadFull::get_quadratic_weights, bp::return_internal_reference<>()),
          bp::make_function(&ActivationModelQuadFull::set_quadratic_weights), "matrix of weights");
}

}  // namespace python
}  // namespace crocoddyl
