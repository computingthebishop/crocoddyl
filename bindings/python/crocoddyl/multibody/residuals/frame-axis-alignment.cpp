///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-axis-alignment.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFrameAxisAlignment() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelFrameAxisAlignment> >();

  bp::class_<ResidualModelFrameAxisAlignment, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFrameAxisAlignment",
      "This residual function defines the tracking of theframe placement residual as r = p - pref, with p and pref "
      "as\n"
      "the current and reference frame placements, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, pinocchio::SE3, std::size_t>(
          bp::args("self", "state", "id", "pref", "nu"),
          "Initialize the frame placement residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param pref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, pinocchio::SE3>(
          bp::args("self", "state", "id", "pref"),
          "Initialize the frame placement residual model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param pref: reference frame placement"))
      .def<void (ResidualModelFrameAxisAlignment::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrameAxisAlignment::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame placement residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelFrameAxisAlignment::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrameAxisAlignment::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrameAxisAlignment::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame placement residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelFrameAxisAlignment::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFrameAxisAlignment::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame placement residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the frame placement residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFrameAxisAlignment::get_id, &ResidualModelFrameAxisAlignment::set_id,
                    "reference frame id")
      .add_property("reference",
                    bp::make_function(&ResidualModelFrameAxisAlignment::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelFrameAxisAlignment::set_reference, "reference frame placement");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataFrameAxisAlignment> >();

  bp::class_<ResidualDataFrameAxisAlignment, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFrameAxisAlignment", "Data for frame placement residual.\n\n",
      bp::init<ResidualModelFrameAxisAlignment*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame placement residual data.\n\n"
          ":param model: frame placement residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("rMf",
                    bp::make_getter(&ResidualDataFrameAxisAlignment::rMf, bp::return_value_policy<bp::return_by_value>()),
                    "error frame placement of the frame")
      .add_property("rJf", bp::make_getter(&ResidualDataFrameAxisAlignment::rJf, bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf", bp::make_getter(&ResidualDataFrameAxisAlignment::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl
