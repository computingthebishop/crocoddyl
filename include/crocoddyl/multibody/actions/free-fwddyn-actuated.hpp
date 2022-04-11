///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_ACTUATED_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_ACTUATED_HPP_

#include <stdexcept>

#ifdef PINOCCHIO_WITH_CPPAD_SUPPORT  // TODO(cmastalli): Removed after merging Pinocchio v.2.4.8
#include <pinocchio/codegen/cppadcg.hpp>
#endif

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for free forward dynamics in multibody systems.
 *
 * This class implements free forward dynamics, i.e.,
 * \f[
 * \mathbf{M}\dot{\mathbf{v}} + \mathbf{h}(\mathbf{q},\mathbf{v}) = \boldsymbol{\tau},
 * \f]
 * where \f$\mathbf{q}\in Q\f$, \f$\mathbf{v}\in\mathbb{R}^{nv}\f$ are the configuration point and generalized velocity
 * (its tangent vector), respectively; \f$\boldsymbol{\tau}\f$ is the torque inputs and
 * \f$\mathbf{h}(\mathbf{q},\mathbf{v})\f$ are the Coriolis effect and gravity field.
 *
 * The derivatives of the system acceleration is computed efficiently based on the analytical derivatives of Articulate
 * Body Algorithm (ABA) as described in \cite carpentier-rss18.
 *
 * The stack of cost functions is implemented in `CostModelSumTpl`. The computation of the free forward dynamics and
 * its derivatives are carrying out inside `calc()` and `calcDiff()` functions, respectively. It is also important to
 * remark that `calcDiff()` computes the derivatives using the latest stored values by `calc()`. Thus, we need to run
 * `calc()` first.
 *
 * \sa `DifferentialActionModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelFreeFwdDynamicsActuatedTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataFreeFwdDynamicsTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelFreeFwdDynamicsActuatedTpl(boost::shared_ptr<StateMultibody> state,
                                            boost::shared_ptr<ActuationModelAbstract> actuation,
                                            boost::shared_ptr<CostModelSum> costs);
  virtual ~DifferentialActionModelFreeFwdDynamicsActuatedTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the free forward-dynamics.
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the free forward-dynamics data
   *
   * @return free forward-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Check that the given data belongs to the free forward-dynamics data
   */
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  /**
   * @brief @copydoc Base::quasiStatic()
   */
  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Return the actuation model
   */
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /**
   * @brief Return the cost model
   */
  const boost::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the Pinocchio model
   */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Return the armature vector
   */
  const VectorXs& get_armature() const;

  /**
   * @brief Modify the armature vector
   */
  void set_armature(const VectorXs& armature);

  /**
   * @brief Print relevant information of the free forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  boost::shared_ptr<ActuationModelAbstract> actuation_;  //!< Actuation model
  boost::shared_ptr<CostModelSum> costs_;                //!< Cost model
  pinocchio::ModelTpl<Scalar>& pinocchio_;               //!< Pinocchio model
  bool without_armature_;                                //!< Indicate if we have defined an armature
  VectorXs armature_;                                    //!< Armature vector
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/free-fwddyn-actuated.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_
