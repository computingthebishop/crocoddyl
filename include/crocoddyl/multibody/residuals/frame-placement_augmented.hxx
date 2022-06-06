///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/frame-placement_augmented.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFramePlacementAugmentedTpl<Scalar>::ResidualModelFramePlacementAugmentedTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const pinocchio::FrameIndex id, const SE3& pref,
                                                                       const std::size_t nu)
    : Base(state, 6, nu, true, false, false),
      id_(id),
      pref_(pref),
      oMf_inv_(pref.inverse()),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFramePlacementAugmentedTpl<Scalar>::ResidualModelFramePlacementAugmentedTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const pinocchio::FrameIndex id, const SE3& pref)
    : Base(state, 6, true, false, false),
      id_(id),
      pref_(pref),
      oMf_inv_(pref.inverse()),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFramePlacementAugmentedTpl<Scalar>::~ResidualModelFramePlacementAugmentedTpl() {}

template <typename Scalar>
void ResidualModelFramePlacementAugmentedTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rMf = oMf_inv_ * d->pinocchio->oMf[id_];
  data->r = pinocchio::log6(d->rMf).toVector();
}

template <typename Scalar>
void ResidualModelFramePlacementAugmentedTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  data->Rx.leftCols(nv-nu_).noalias() = d->rJf * d->fJf;

  // Eigen::MatrixXd tempfJf(6, nv - 4);
  // pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, tempfJf);
  // d->fJf.leftCols(nv - 4) = tempfJf;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFramePlacementAugmentedTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ResidualModelFramePlacementAugmentedTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
  typename SE3::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, pref_.rotation());
  os << "ResidualModelFramePlacement {frame=" << pin_model_->frames[id_].name
     << ", tref=" << pref_.translation().transpose().format(fmt) << ", qref=" << qref.coeffs().transpose().format(fmt)
     << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFramePlacementAugmentedTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::SE3Tpl<Scalar>& ResidualModelFramePlacementAugmentedTpl<Scalar>::get_reference() const {
  return pref_;
}

template <typename Scalar>
void ResidualModelFramePlacementAugmentedTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFramePlacementAugmentedTpl<Scalar>::set_reference(const SE3& placement) {
  pref_ = placement;
  oMf_inv_ = placement.inverse();
}

}  // namespace crocoddyl
