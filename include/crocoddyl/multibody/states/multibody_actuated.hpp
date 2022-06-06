///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_MULTIBODY_ACTUATED_HPP_
#define CROCODDYL_MULTIBODY_STATES_MULTIBODY_ACTUATED_HPP_

#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/state-base.hpp"

#include <math.h>  

namespace crocoddyl {

/**
 * @brief State multibody representation
 *
 * A multibody state is described by the configuration point and its tangential velocity, or in other words, by the
 * generalized position and velocity coordinates of a rigid-body system. For this state, we describe its operators:
 * difference, integrates, transport and their derivatives for any Pinocchio model.
 *
 * For more details about these operators, please read the documentation of the `StateAbstractTpl` class.
 *
 * \sa `diff()`, `integrate()`, `Jdiff()`, `Jintegrate()` and `JintegrateTransport()`
 */
template <typename _Scalar>
class StateMultibodyActuatedTpl : public StateMultibodyTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> Base;
  typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the multibody state
   *
   * @param[in] model  Pinocchio model
   */
  explicit StateMultibodyActuatedTpl(boost::shared_ptr<PinocchioModel> model, std::size_t nr);
  StateMultibodyActuatedTpl();
  virtual ~StateMultibodyActuatedTpl();

  /**
   * @brief Generate a zero state.
   *
   * Note that the zero configuration is computed using `pinocchio::neutral`.
   */
  virtual VectorXs zero() const;

  /**
   * @brief Generate a random state
   *
   * Note that the random configuration is computed using `pinocchio::random` which satisfies the manifold definition
   * (e.g., the quaterion definition)
   */
  virtual VectorXs rand() const;

  virtual void diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const;
  virtual void integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const;
  virtual void Jdiff(const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&, Eigen::Ref<MatrixXs> Jfirst,
                     Eigen::Ref<MatrixXs> Jsecond, const Jcomponent firstsecond = both) const;

  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both, const AssignmentOp = setto) const;
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const;

  /**
   * @brief Return the Pinocchio model (i.e., model of the rigid body system)
   */
  const boost::shared_ptr<PinocchioModel>& get_pinocchio() const;

  /**
   * @brief Return the number of rotors
   */
  std::size_t get_nrotors() const;

 protected:
  using Base::has_limits_;
  using Base::lb_;
  using Base::ndx_;
  using Base::nq_;
  using Base::nv_;
  using Base::nx_;
  using Base::ub_;

 private:
  boost::shared_ptr<PinocchioModel> pinocchio_;  //!< Pinocchio model
  VectorXs x0_;                                  //!< Zero state
  std::size_t nr_;                               //!< number of rotors
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/states/multibody_actuated.hxx"

#endif  // CROCODDYL_MULTIBODY_STATES_MULTIBODY_ACTUATED_HPP_
