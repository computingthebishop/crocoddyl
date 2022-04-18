///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn-actuated.hpp"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/cholesky.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::DifferentialActionModelFreeFwdDynamicsActuatedTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs)
    : Base(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      costs_(costs),
      pinocchio_(*state->get_pinocchio().get()),
      without_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())) {
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
  n_rotors_ = nu_;
}

template <typename Scalar>
DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::~DifferentialActionModelFreeFwdDynamicsActuatedTpl() {}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq()-(2*n_rotors_));
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),state_->get_nv()-n_rotors_);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> rotors_v = x.tail(n_rotors_);
  // std::cout << "rotor speeds: " << x.tail(n_rotors_).transpose() << "\n";
  // std::cout << "Control action: " << u.transpose() << "\n";
  actuation_->calc(d->multibody.actuation, x, u);
  Scalar time_ct = 0.1; // TODO this as a parameter
  // Computing the dynamics using ABA or manually for armature case
  if (without_armature_) {
    try
    {
      d->xout.head(state_->get_nv()-n_rotors_) = pinocchio::aba(pinocchio_, d->pinocchio, q, v, d->multibody.actuation->tau);
    }
    catch(const std::exception& e)
    {
      std::cerr << "Error running ABA" << '\n';
      std::cerr << e.what() << '\n';
    }
    // compute FO system acceleration
    // d->xout.tail(n_rotors_) = VectorXs::Zero(n_rotors_); 
    d->xout.tail(n_rotors_) = ((-rotors_v)/time_ct) + ((1/time_ct)*u);
    pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
  } else {
    std::cerr << "[DifferentialActionModelFreeFwdDynamicsActuatedTpl] Error armature not implemented" << '\n';
    exit(1);
    pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
    d->pinocchio.M.diagonal() += armature_;
    pinocchio::cholesky::decompose(pinocchio_, d->pinocchio);
    d->Minv.setZero();
    pinocchio::cholesky::computeMinv(pinocchio_, d->pinocchio, d->Minv);
    d->u_drift = d->multibody.actuation->tau - d->pinocchio.nle;
    d->xout.noalias() = d->Minv * d->u_drift;
  }

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq()-(2*n_rotors_));
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),state_->get_nv()-n_rotors_);

  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);

  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
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
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq()-(2*n_rotors_));
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),state_->get_nv()-n_rotors_);

  Data* d = static_cast<Data*>(data.get());

  actuation_->calcDiff(d->multibody.actuation, x, u);
  Scalar time_ct = 0.1; // TODO this as a parameter

  // Computing the dynamics derivatives  
  if (without_armature_) {
    Eigen::MatrixXd tempFx(nv-n_rotors_,(nv-n_rotors_)*2); //create temporal jacobian of the dinamics to obtain values from pinocchio::computeABADerivatives
    //dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()) is initialized in ActuationModelAbstractTpl
    Eigen::MatrixXd temp_dtau_dx(nv-n_rotors_,state_->get_ndx()-(n_rotors_*2)); //create temporal jacobian of the dinamics to compute partially Fu

    pinocchio::computeABADerivatives(pinocchio_, d->pinocchio, q, v, d->multibody.actuation->tau, tempFx.leftCols(nv-n_rotors_),
                                     tempFx.rightCols(nv-n_rotors_), d->pinocchio.Minv); 
                                     
    // assing values of d->multibody.actuation->dtau_dx to the temporal matrix
    temp_dtau_dx.leftCols(nv-n_rotors_) = d->multibody.actuation->dtau_dx.block(0,0,nv-n_rotors_,nv-n_rotors_);
    temp_dtau_dx.rightCols(nv-n_rotors_) = d->multibody.actuation->dtau_dx.block(0,nv-n_rotors_,nv-n_rotors_,nv-n_rotors_);

    // compute partial Fx
    tempFx += d->pinocchio.Minv * temp_dtau_dx;
    // Assign partial Fx to d->Fx section
    // derivatives of drone dynamics 
    d->Fx.block(0,0,nv-n_rotors_,nv-n_rotors_) = tempFx.leftCols(nv-n_rotors_); //first block containing derivatives wrt position + orientation 
    d->Fx.block(0,nv-n_rotors_,nv-n_rotors_,n_rotors_) = MatrixXs::Zero(nv-n_rotors_,n_rotors_); //second block containing deivatives wrt rotor position 
    d->Fx.block(0,nv,nv-n_rotors_,nv-n_rotors_) = tempFx.rightCols(nv-n_rotors_); //third block containing deivatives wrt linear + angular velocities 
    d->Fx.block(0,(2*nv)-n_rotors_,nv-n_rotors_,n_rotors_) = MatrixXs::Zero(nv-n_rotors_,n_rotors_); //fourth block containing deivatives wrt rotor velocities 
    // derivatives of actuator dynamics 
    d->Fx.block(nv-n_rotors_,0,n_rotors_,nv-n_rotors_) = MatrixXs::Zero(n_rotors_,nv-n_rotors_); //first block containing derivatives wrt position + orientation 
    d->Fx.block(nv-n_rotors_,nv-n_rotors_,n_rotors_,n_rotors_) = MatrixXs::Zero(n_rotors_,n_rotors_); //second block containing derivatives wrt rotor position 
    d->Fx.block(nv-n_rotors_,nv,n_rotors_,nv-n_rotors_) = MatrixXs::Zero(n_rotors_,nv-n_rotors_); //third block containing deivatives wrt linear + angular velocities 
    //d->Fx.block(nv-n_rotors_,(2*nv)-n_rotors_,n_rotors_,n_rotors_) = MatrixXs::Zero(n_rotors_,n_rotors_); //fourth block containing deivatives wrt rotor velocities 
    d->Fx.block(nv-n_rotors_,(2*nv)-n_rotors_,n_rotors_,n_rotors_).diagonal().array() = (Scalar)(-1/time_ct); //fourth block containing deivatives wrt rotor velocities 

    d->Fu.topRows(nv-n_rotors_) = d->pinocchio.Minv * d->multibody.actuation->dtau_du;
    //d->Fu.bottomRows(n_rotors_) = MatrixXs::Zero(n_rotors_,n_rotors_);
    d->Fu.bottomRows(n_rotors_).diagonal().array() = (Scalar)(1/time_ct);
  } else {
    std::cerr << "[DifferentialActionModelFreeFwdDynamicsActuatedTpl] Error armature not implemented" << '\n';
    exit(1);
    pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout);
    d->dtau_dx.leftCols(nv) = d->multibody.actuation->dtau_dx.leftCols(nv) - d->pinocchio.dtau_dq;
    d->dtau_dx.rightCols(nv) = d->multibody.actuation->dtau_dx.rightCols(nv) - d->pinocchio.dtau_dv;
    d->Fx.noalias() = d->Minv * d->dtau_dx;
    d->Fu.noalias() = d->Minv * d->multibody.actuation->dtau_du;
  }

  // Computing the cost derivatives
  costs_->calcDiff(d->costs, x, u);
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());

  costs_->calcDiff(d->costs, x);
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}
template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t, const Scalar) {
  //TODO implement function
  std::cerr << "[DifferentialActionModelFreeFwdDynamicsActuatedTpl] Error quasiStatic not implemented" << '\n';
  exit(1);
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  // Static casting the data
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();

  // Check the velocity input is zero
  assert_pretty(x.tail(nv).isZero(), "The velocity input should be zero for quasi-static to work.");

  d->tmp_xstatic.head(nq) = q; 
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv), d->tmp_xstatic.tail(nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);

  u.noalias() = pseudoInverse(d->multibody.actuation->dtau_du) * d->pinocchio.tau;
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelFreeFwdDynamics {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx()
     << ", nu=" << nu_ << "}";
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
void DifferentialActionModelFreeFwdDynamicsActuatedTpl<Scalar>::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
  }

  armature_ = armature;
  without_armature_ = false;
}

}  // namespace crocoddyl
