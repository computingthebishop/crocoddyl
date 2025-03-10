///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, IRI: CSIC-UPC, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_FOS_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_FOS_HPP_

#include <iostream>
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/** TODO
 * @brief Multicopter actuation model
 *
 * This actuation model is aimed for those robots whose base_link is actuated using a propulsion system, e.g.,
 * a multicopter or an aerial manipulator (multicopter with a robotic arm attached).
 * Control input: the thrust (force) created by each propeller.
 * tau_f matrix: this matrix relates the thrust of each propeller to the net force and torque that it causes to the
 * base_link. For a simple quadrotor: tau_f.nrows = 6, tau_f.ncols = 4
 *
 * Both actuation and Jacobians are computed analytically by `calc` and `calcDiff`, respectively.
 *
 * Reference: M. Geisert and N. Mansard, "Trajectory generation for quadrotor based systems using numerical optimal
 * control," 2016 IEEE International Conference on Robotics and Automation (ICRA), Stockholm, 2016, pp. 2958-2964. See
 * Section III.C.
 *
 * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelMultiCopterBaseFosTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  /**
   * @brief Initialize the multicopter actuation model
   *
   * @param[in] state  State of the dynamical system
   * @param[in] tau_f  Matrix that maps the thrust of each propeller to the net force and torque
   */
  ActuationModelMultiCopterBaseFosTpl(boost::shared_ptr<StateMultibody> state, const Eigen::Ref<const Matrix6xs>& tau_f)
      : Base(state, tau_f.cols()), n_rotors_(tau_f.cols()) {
    pinocchio::JointModelFreeFlyerTpl<Scalar> ff_joint;
    if (state->get_pinocchio()->joints[1].shortname() != ff_joint.shortname()) {
      throw_pretty("Invalid argument: "
                   << "the first joint has to be free-flyer");
    }

    tau_f_ = MatrixXs::Zero(state_->get_nv()-nu_, nu_); // tau matrix defines generalized forces based on control inputs
    tau_f_.block(0, 0, 6, n_rotors_) = tau_f;
    if (nu_ > n_rotors_) {
      tau_f_.bottomRightCorner(nu_ - n_rotors_, nu_ - n_rotors_).diagonal().setOnes();
    }
  }

  DEPRECATED("Use constructor without n_rotors",
             ActuationModelMultiCopterBaseFosTpl(boost::shared_ptr<StateMultibody> state, const std::size_t n_rotors,
                                              const Eigen::Ref<const Matrix6xs>& tau_f));
  virtual ~ActuationModelMultiCopterBaseFosTpl() {}

  virtual void calc(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    //std::cout << "rotor speeds: " << x.tail(n_rotors_).transpose() << "\n";
    // data->tau.noalias() = tau_f_ * x.tail(n_rotors_);
    Eigen::VectorXd debug_r_velocities(n_rotors_);
    // debug_r_velocities = (x.tail(n_rotors_).array()*x.tail(n_rotors_).array());
    for (std::size_t i = 0; i < n_rotors_; i++)
    {
      if (x(i+state_->get_nq()+state_->get_nv()-n_rotors_) >= 0) 
        debug_r_velocities(i) = x(i+state_->get_nq()+state_->get_nv()-n_rotors_) * x(i+state_->get_nq()+state_->get_nv()-n_rotors_);
      else
        debug_r_velocities(i) = -(x(i+state_->get_nq()+state_->get_nv()-n_rotors_) * x(i+state_->get_nq()+state_->get_nv()-n_rotors_));
    }
    
    data->tau.noalias() = tau_f_ * debug_r_velocities; 
    //data->tau.noalias() = tau_f_ * u;
  }

  virtual void calcDiff(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>&) {
    // The derivatives has constant values which were set in createData.
    //assert_pretty(MatrixXs(data->dtau_du).isApprox(tau_f_), "dtau_du has wrong value");
    for (std::size_t i = 0; i < n_rotors_; i++)
    {
      if (x(i+state_->get_nq()+state_->get_nv()-n_rotors_) >= 0) 
        data->dtau_dx.col(i+state_->get_ndx()-n_rotors_) = tau_f_.col(i)*2*x(i+state_->get_nq()+state_->get_nv()-n_rotors_);
      else
        data->dtau_dx.col(i+state_->get_ndx()-n_rotors_) = -tau_f_.col(i)*2*x(i+state_->get_nq()+state_->get_nv()-n_rotors_);
    }
  }

  boost::shared_ptr<Data> createData() {
    boost::shared_ptr<Data> data = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    //data->dtau_du = tau_f_;
    data->dtau_du = MatrixXs::Zero(6,n_rotors_); // derivative wrt controls is 0 as it depends on the state only
    data->dtau_dx.resize(state_->get_nv()-n_rotors_,state_->get_ndx()); //required resize as the ActuationDataAbstractTpl initiliases the matrix based on state->nv
    data->dtau_dx.setZero();
    data->dtau_dx.rightCols(n_rotors_) = tau_f_;
    return data;
  }

  std::size_t get_nrotors() const { return n_rotors_; };
  const MatrixXs& get_tauf() const { return tau_f_; };
  void set_tauf(const Eigen::Ref<const MatrixXs>& tau_f) { tau_f_ = tau_f; }

 protected:
  // Specific of multicopter
  MatrixXs tau_f_;  // Matrix from rotors thrust to body force/moments
  std::size_t n_rotors_;

  using Base::nu_;
  using Base::state_;
};

template <typename Scalar>
ActuationModelMultiCopterBaseFosTpl<Scalar>::ActuationModelMultiCopterBaseFosTpl(boost::shared_ptr<StateMultibody> state,
                                                                           const std::size_t n_rotors,
                                                                           const Eigen::Ref<const Matrix6xs>& tau_f)
    : Base(state, state->get_nv() - 6 + n_rotors), n_rotors_(n_rotors) {
  pinocchio::JointModelFreeFlyerTpl<Scalar> ff_joint;
  if (state->get_pinocchio()->joints[1].shortname() != ff_joint.shortname()) {
    throw_pretty("Invalid argument: "
                 << "the first joint has to be free-flyer");
  }

  tau_f_ = MatrixXs::Zero(state_->get_nv(), nu_);
  tau_f_.block(0, 0, 6, n_rotors_) = tau_f;
  if (nu_ > n_rotors_) {
    tau_f_.bottomRightCorner(nu_ - n_rotors_, nu_ - n_rotors_).diagonal().setOnes();
  }
  std::cerr << "Deprecated ActuationModelMultiCopterBase: Use constructor without n_rotors." << std::endl;
}

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_MULTICOPTER_BASE_HPP_
