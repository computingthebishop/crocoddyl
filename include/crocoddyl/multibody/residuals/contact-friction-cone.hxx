///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelContactFrictionConeTpl<Scalar>::ResidualModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const FrictionCone& fref,
    const std::size_t nu)
    : Base(state, fref.get_nf() + 1, nu, true, true, true), id_(id), fref_(fref) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <= id) {
    throw_pretty("Invalid argument: "
                 << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelContactFrictionConeTpl<Scalar>::ResidualModelContactFrictionConeTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const FrictionCone& fref)
    : Base(state, fref.get_nf() + 1), id_(id), fref_(fref) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <= id) {
    throw_pretty("Invalid argument: "
                 << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelContactFrictionConeTpl<Scalar>::~ResidualModelContactFrictionConeTpl() {}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                       const Eigen::Ref<const VectorXs>&,
                                                       const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual of the friction cone. Note that we need to transform the force
  // to the contact frame
  data->r.noalias() = fref_.get_A() * d->contact->jMf.actInv(d->contact->f).linear();
}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                       const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                           const Eigen::Ref<const VectorXs>&,
                                                           const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const MatrixX3s& A = fref_.get_A();

  switch (d->contact_type) {
    case Contact2D: {
      // Valid for xz plane
      data->Rx.noalias() = A.col(0) * df_dx.row(0) + A.col(2) * df_dx.row(1);
      data->Ru.noalias() = A.col(0) * df_du.row(0) + A.col(2) * df_du.row(1);
      break;
    }
    case Contact3D:
      data->Rx.noalias() = A * df_dx;
      data->Ru.noalias() = A * df_du;
      break;
    case Contact6D:
      data->Rx.noalias() = A * df_dx.template topRows<3>();
      data->Ru.noalias() = A * df_du.template topRows<3>();
      break;
    default:
      break;
  }
}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                           const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelContactFrictionConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::print(std::ostream& os) const {
  boost::shared_ptr<StateMultibody> s = boost::static_pointer_cast<StateMultibody>(state_);
  os << "ResidualModelContactFrictionCone {frame=" << s->get_pinocchio()->frames[id_].name << ", mu=" << fref_.get_mu()
     << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactFrictionConeTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const FrictionConeTpl<Scalar>& ResidualModelContactFrictionConeTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactFrictionConeTpl<Scalar>::set_reference(const FrictionCone& reference) {
  fref_ = reference;
}

}  // namespace crocoddyl
