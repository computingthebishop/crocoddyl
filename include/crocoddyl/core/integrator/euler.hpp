///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
#define CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class IntegratedActionModelEulerTpl : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataEulerTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar> DifferentialActionModelAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> ControlParametrizationModelAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedActionModelEulerTpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  IntegratedActionModelEulerTpl(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                boost::shared_ptr<ControlParametrizationModelAbstract> control,
                                const Scalar time_step = Scalar(1e-3), const bool with_cost_residual = true);
  virtual ~IntegratedActionModelEulerTpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Print relevant information of the Euler integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::control_;             //!< Control parametrization
  using Base::differential_;        //!< Differential action model
  using Base::enable_integration_;  //!< False for the terminal horizon node, where integration is not needed
  using Base::nu_;                  //!< Dimension of the control
  using Base::state_;               //!< Model of the state
  using Base::time_step2_;          //!< Square of the time step used for integration
  using Base::time_step_;           //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual is used
};

template <typename _Scalar>
struct IntegratedActionDataEulerTpl : public IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataEulerTpl(Model<Scalar>* const model) : Base(model) {
    differential = model->get_differential()->createData();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nv = model->get_state()->get_nv();
    dx = VectorXs::Zero(ndx);
    da_du = MatrixXs::Zero(nv, model->get_nu());
    Lwu = MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu());
  }
  virtual ~IntegratedActionDataEulerTpl() {}

  boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > differential;
  VectorXs dx;
  MatrixXs da_du;
  MatrixXs Lwu;  // Hessian of the cost function with respect to the control input (w) and control parameters (u)

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/integrator/euler.hxx"

#endif  // CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
