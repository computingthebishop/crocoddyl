///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/actuations/multicopter-base-fos.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationModelMultiCopterBaseFos() {
  bp::register_ptr_to_python<boost::shared_ptr<crocoddyl::ActuationModelMultiCopterBaseFos> >();

  bp::class_<ActuationModelMultiCopterBaseFos, bp::bases<ActuationModelAbstract> >(
      "ActuationModelMultiCopterBaseFos",
      "Actuation models with base actuated by several propellers (e.g. aerial manipulators).",
      bp::init<boost::shared_ptr<StateMultibody>, Eigen::Matrix<double, 6, Eigen::Dynamic>, double >(
          bp::args("self", "state", "tau_f", "cf"),
          "Initialize the full actuation model.\n\n"
          ":param state: state of multibody system\n"
          ":param tau_f: matrix that maps rotors thrust to generalized torque of the flying base."))
      .def(bp::init<boost::shared_ptr<StateMultibody>, std::size_t, Eigen::Matrix<double, 6, Eigen::Dynamic> >(
          bp::args("self", "state", "nrotors", "tau_f"),
          "Initialize the full actuation model.\n\n"
          ":param state: state of multibody system, \n"
          ":param nrotors: number of rotors of the flying base, \n"
          ":param tau_f: matrix that maps rotors thrust to generalized torque of the flying base."))
      .def<void (ActuationModelMultiCopterBaseFos::*)(const boost::shared_ptr<ActuationDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActuationModelMultiCopterBaseFos::calc, bp::args("self", "data", "x", "u"),
          "Compute the actuation signal from the control input u.\n\n"
          ":param data: multicopter-base actuation data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def("calcDiff", &ActuationModelMultiCopterBaseFos::calcDiff, bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the full actuation. It assumes that calc\n"
           "has been run first. The reason is that the derivatives are constant and\n"
           "defined in createData. The Hessian is constant, so we don't write again this value.\n"
           ":param data: multicopter-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def("createData", &ActuationModelMultiCopterBaseFos::createData, bp::args("self"),
           "Create the multicopter-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("nrotors", bp::make_function(&ActuationModelMultiCopterBaseFos::get_nrotors),
                    "Number of rotors in the flying base")
      .add_property(
          "tauf",
          bp::make_function(&ActuationModelMultiCopterBaseFos::get_tauf, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ActuationModelMultiCopterBaseFos::set_tauf), "Matrix mapping from thrusts to body torque");
}

}  // namespace python
}  // namespace crocoddyl
