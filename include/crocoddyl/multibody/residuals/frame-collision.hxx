///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/frame-collision.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameCollisionTpl<Scalar>::ResidualModelFrameCollisionTpl(boost::shared_ptr<StateMultibody> state,
                                                                       const pinocchio::FrameIndex id,
                                                                       const Vector3s& xref, const std::size_t nu,
                                                                       const Scalar eps)
    : Base(state, 1, nu, true, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()), eps_(eps) {}

template <typename Scalar>
ResidualModelFrameCollisionTpl<Scalar>::~ResidualModelFrameCollisionTpl() {}

template <typename Scalar>
void ResidualModelFrameCollisionTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>&,
                                                  const Eigen::Ref<const VectorXs>&) {
  // Compute the frame translation w.r.t. the reference frame
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->t = d->pinocchio->oMf[id_].translation() - xref_;
  using std::sqrt;
  d->r(0) = sqrt(d->t.squaredNorm() + eps_);
}

template <typename Scalar>
void ResidualModelFrameCollisionTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame translation
  const std::size_t nv = state_->get_nv();
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  d->fJt = d->t / d->r(0);
  d->Rx.leftCols(nv).noalias() = d->fJt.transpose() * d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFrameCollisionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ResidualModelFrameCollisionTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
  os << "ResidualModelFrameCollision {frame=" << pin_model_->frames[id_].name
     << ", tref=" << xref_.transpose().format(fmt) << ", eps=" << eps_ << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameCollisionTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& ResidualModelFrameCollisionTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
void ResidualModelFrameCollisionTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameCollisionTpl<Scalar>::set_reference(const Vector3s& translation) {
  xref_ = translation;
}

}  // namespace crocoddyl
