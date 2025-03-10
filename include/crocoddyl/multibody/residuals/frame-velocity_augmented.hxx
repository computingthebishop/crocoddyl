///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include "crocoddyl/multibody/residuals/frame-velocity_augmented.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameVelocityAugmentedTpl<Scalar>::ResidualModelFrameVelocityAugmentedTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Motion& velocity,
                                                                     const pinocchio::ReferenceFrame type,
                                                                     const std::size_t nu)
    : Base(state, 6, nu, true, true, false),
      id_(id),
      vref_(velocity),
      type_(type),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFrameVelocityAugmentedTpl<Scalar>::ResidualModelFrameVelocityAugmentedTpl(boost::shared_ptr<StateMultibody> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Motion& velocity,
                                                                     const pinocchio::ReferenceFrame type)
    : Base(state, 6, true, true, false), id_(id), vref_(velocity), type_(type), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFrameVelocityAugmentedTpl<Scalar>::~ResidualModelFrameVelocityAugmentedTpl() {}

template <typename Scalar>
void ResidualModelFrameVelocityAugmentedTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>&,
                                                 const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  data->r = (pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_, type_) - vref_).toVector();
}

template <typename Scalar>
void ResidualModelFrameVelocityAugmentedTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>&,
                                                     const Eigen::Ref<const VectorXs>&) {
  // Get the partial derivatives of the local frame velocity
  Data* d = static_cast<Data*>(data.get());
  const std::size_t s_nv = state_->get_nv();
  const std::size_t p_nv = pin_model_->nv;
  // pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio, id_, type_, data->Rx.leftCols(nv),
  //                                        data->Rx.rightCols(nv));
  pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio, id_, type_, data->Rx.block(0,0,p_nv,p_nv),
                                         data->Rx.block(0,s_nv,p_nv,p_nv));
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFrameVelocityAugmentedTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ResidualModelFrameVelocityAugmentedTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
  os << "ResidualModelFrameVelocity {frame=" << pin_model_->frames[id_].name
     << ", vref=" << vref_.toVector().transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameVelocityAugmentedTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::MotionTpl<Scalar>& ResidualModelFrameVelocityAugmentedTpl<Scalar>::get_reference() const {
  return vref_;
}

template <typename Scalar>
pinocchio::ReferenceFrame ResidualModelFrameVelocityAugmentedTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void ResidualModelFrameVelocityAugmentedTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameVelocityAugmentedTpl<Scalar>::set_reference(const Motion& velocity) {
  vref_ = velocity;
}

template <typename Scalar>
void ResidualModelFrameVelocityAugmentedTpl<Scalar>::set_type(const pinocchio::ReferenceFrame type) {
  type_ = type;
}

}  // namespace crocoddyl
