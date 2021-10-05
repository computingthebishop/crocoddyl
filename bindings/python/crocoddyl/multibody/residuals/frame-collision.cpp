///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-collision.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameCollision() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelFrameCollision>>();

  bp::class_<ResidualModelFrameCollision, bp::bases<ResidualModelAbstract>>(
      "ResidualModelFrameCollision",
      "This residual function defines the smooth 2-norm of the Euclidean distance between\n"
      "a frame and an external object." bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                                                 Eigen::Vector3d, std::size_t, bp::optional<double>>(
          bp::args("self", "state", "id", "xref", "nu", "eps"),
          "Initialize the frame translation residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector"
          ":param eps: smoothing factor"))
      .def<void (ResidualModelFrameCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrameCollision::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame translation residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFrameCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrameCollision::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame translation residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFrameCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameCollision::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame translation residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the frame translation residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameCollision::get_id, &ResidualModelFrameCollision::set_id,
                    "reference frame id")
      .add_property("reference",
                    bp::make_function(&ResidualModelFrameCollision::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelFrameCollision::set_reference, "reference frame translation");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataFrameCollision>>();

  bp::class_<ResidualDataFrameCollision, bp::bases<ResidualDataAbstract>>(
      "ResidualDataFrameCollision", "Data for frame translation residual.\n\n",
      bp::init<ResidualModelFrameCollision*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame translation residual data.\n\n"
          ":param model: frame translation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameCollision::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("t", bp::make_getter(&ResidualDataFrameCollision::t, bp::return_internal_reference<>()),
                    "Euclidean distance")
      .add_property("fJf", bp::make_getter(&ResidualDataFrameCollision::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame")
      .add_property("fJt", bp::make_getter(&ResidualDataFrameCollision::fJt, bp::return_internal_reference<>()),
                    "Jacobian of the smooth 2-norm w.r.t vector");
}

}  // namespace python
}  // namespace crocoddyl
