///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::IntegratedActionModelEulerTpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model,
    boost::shared_ptr<ControlParametrizationModelAbstract> control, const Scalar time_step,
    const bool with_cost_residual)
    : Base(model, control, time_step, with_cost_residual) {}

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::IntegratedActionModelEulerTpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model, const Scalar time_step, const bool with_cost_residual)
    : Base(model, time_step, with_cost_residual) {}

template <typename Scalar>
IntegratedActionModelEulerTpl<Scalar>::~IntegratedActionModelEulerTpl() {}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x,
                                                 const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  const std::size_t nv = differential_->get_state()->get_nv();
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  control_->calc(d->control, 0., u);
  differential_->calc(d->differential, x, d->control->w);
  const VectorXs& a = d->differential->xout;
  d->dx.head(nv).noalias() = v * time_step_ + a * time_step2_;
  d->dx.tail(nv).noalias() = a * time_step_;
  differential_->get_state()->integrate(x, d->dx, d->xnext);
  d->cost = time_step_ * d->differential->cost;
  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  differential_->calc(d->differential, x);
  d->dx.setZero();
  d->cost = d->differential->cost;
  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x,
                                                     const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());

  control_->calc(d->control, 0., u);
  differential_->calcDiff(d->differential, x, d->control->w);
  const MatrixXs& da_dx = d->differential->Fx;
  const MatrixXs& da_du = d->differential->Fu; // this is [6 4], should be [10 4] 
  control_->multiplyByJacobian(d->control, da_du, d->da_du);
  d->Fx.topRows(nv).noalias() = da_dx * time_step2_;
  d->Fx.bottomRows(nv).noalias() = da_dx * time_step_;
  d->Fx.topRightCorner(nv, nv).diagonal().array() += Scalar(time_step_);
  d->Fu.topRows(nv).noalias() = time_step2_ * d->da_du;
  d->Fu.bottomRows(nv).noalias() = time_step_ * d->da_du;
  state_->JintegrateTransport(x, d->dx, d->Fx, second);
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
  state_->JintegrateTransport(x, d->dx, d->Fu, second);

  d->Lx.noalias() = time_step_ * d->differential->Lx;
  control_->multiplyJacobianTransposeBy(d->control, d->differential->Lu, d->Lu);
  d->Lu *= time_step_;
  d->Lxx.noalias() = time_step_ * d->differential->Lxx;
  control_->multiplyByJacobian(d->control, d->differential->Lxu, d->Lxu);
  d->Lxu *= time_step_;
  control_->multiplyByJacobian(d->control, d->differential->Luu, d->Lwu);
  control_->multiplyJacobianTransposeBy(d->control, d->Lwu, d->Luu);
  d->Luu *= time_step_;
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  differential_->calcDiff(d->differential, x);
  state_->Jintegrate(x, d->dx, d->Fx, d->Fx);
  d->Lx = d->differential->Lx;
  d->Lxx = d->differential->Lxx;
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > IntegratedActionModelEulerTpl<Scalar>::createData() {
  if (control_->get_nu() > differential_->get_nu())
    std::cerr
        << "Warning: It is useless to use an Euler integrator with a control parametrization larger than PolyZero"
        << std::endl;
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool IntegratedActionModelEulerTpl<Scalar>::checkData(const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (data != NULL) {
    return differential_->checkData(d->differential);
  } else {
    return false;
  }
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data,
                                                        Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                                                        const std::size_t maxiter, const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  const boost::shared_ptr<Data>& d = boost::static_pointer_cast<Data>(data);

  d->control->w *= 0.;
  differential_->quasiStatic(d->differential, d->control->w, x, maxiter, tol);
  control_->params(d->control, 0., d->control->w);
  u = d->control->u;
}

template <typename Scalar>
void IntegratedActionModelEulerTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedActionModelEuler {dt=" << time_step_ << ", " << *differential_ << "}";
}

}  // namespace crocoddyl
