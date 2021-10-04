///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/frame-axis-alignment.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameAxisAlignmentTpl<Scalar>::ResidualModelFrameAxisAlignmentTpl(boost::shared_ptr<StateMultibody> state,
                                                                               const pinocchio::FrameIndex id,
                                                                               const Vector3s& axisref,
                                                                               const std::size_t direction,
                                                                               const std::size_t nu)
    : Base(state, 1, nu, true, false, false), id_(id), direction_(direction), pin_model_(state->get_pinocchio()) {
  if (direction_ > 2)
    throw_pretty("Invalid argument: "
                 << "direction (" + std::to_string(direction_) + ") is larger than 2");

  axisref_ = axisref;
  axisref_.normalize();

  switch (direction) {
    case 0:
      v_axis_frame_ = Vector3s::UnitX();
      break;
    case 1:
      v_axis_frame_ = Vector3s::UnitY();
      break;
    case 2:
      v_axis_frame_ = Vector3s::UnitZ();
      break;
  }

  skew_axis_frame_ = pinocchio::skew(v_axis_frame_);
}

template <typename Scalar>
ResidualModelFrameAxisAlignmentTpl<Scalar>::~ResidualModelFrameAxisAlignmentTpl() {}

template <typename Scalar>
void ResidualModelFrameAxisAlignmentTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->axis = d->pinocchio->oMf[id_].rotation() * v_axis_frame_;
  data->r = axisref_.transpose() * d->axis + MatrixXs::Identity(1, 1);
}

template <typename Scalar>
void ResidualModelFrameAxisAlignmentTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                          const Eigen::Ref<const VectorXs>&,
                                                          const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  d->ractJrot = -d->pinocchio->oMf[id_].rotation() * skew_axis_frame_;
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  data->Rx.leftCols(nv).noalias() = axisref_.transpose() * d->ractJrot * d->fJf.block(3, 0, 3, nv);
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFrameAxisAlignmentTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ResidualModelFrameAxisAlignmentTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
  os << "ResidualModelFrameAxisAlignment {frame=" << pin_model_->frames[id_].name
     << ", tref=" << axisref_.transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameAxisAlignmentTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& ResidualModelFrameAxisAlignmentTpl<Scalar>::get_reference() const {
  return axisref_;
}

template <typename Scalar>
void ResidualModelFrameAxisAlignmentTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameAxisAlignmentTpl<Scalar>::set_reference(const Vector3s& reference) {
  axisref_ = reference;
  axisref_.normalize();
}

}  // namespace crocoddyl
