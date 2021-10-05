
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, IRI (CSIC-UPC)
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_TRANSLATION_HPP_

#include <pinocchio/multibody/fwd.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame collision residual
 *
 * This residual function defines the smooth 2norm of the Euclidean distance between a frame and an external object as
 * \f$\mathbf{r}=\sqrt{\epsilon + sum^nr_{i=0} \|r_i\|^2} \f$, where \mathbf{r} = \mathbf{t}-\mathbf{t}^* \f$ and
 * \mathbf{t},\mathbf{t}^*\in~\mathbb{R}^3\f$ are the current and reference frame collisions, respectively. Note that
 * the dimension of the residual vector is 1. Furthermore, the Jacobians of the residual function are computed
 * analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFrameCollisionTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFrameCollisionTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;

  /**
   * @brief Initialize the frame collision residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] xref   Reference frame collision
   * @param[in] nu     Dimension of the control vector
   * @param[in] eps    Smoothing factor (default: 1.)
   */
  ResidualModelFrameCollisionTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex,
                                 const Vector3s& xref, const std::size_t nu, const Scalar eps = Scalar(1.));

  virtual ~ResidualModelFrameCollisionTpl();

  /**
   * @brief Compute the frame collision residual
   *
   * @param[in] data  frame collision residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the frame collision residual
   *
   * @param[in] data  frame collision residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame collision residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference frame collision
   */
  const Vector3s& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference frame collision reference
   */
  void set_reference(const Vector3s& reference);

  /**
   * @brief Print relevant information of the frame collision residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::unone_;
  using Base::v_dependent_;

 private:
  pinocchio::FrameIndex id_;                                              //!< Reference frame id
  Vector3s xref_;                                                         //!< Reference frame collision
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
  Scalar eps_;                                                            //!< Smoothing factor
};

template <typename _Scalar>
struct ResidualDataFrameCollisionTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::Vector3s Vector3s;

  template <template <typename Scalar> class Model>
  ResidualDataFrameCollisionTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), fJf(6, model->get_state()->get_nv()) {
    fJf.setZero();
    fJt.setZero();
    t.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  Matrix6xs fJf;                          //!< Local Jacobian of the frame
  Vector3s fJt;                           //!< Jacobian smooth abs w.r.t. vector
  Vector3s t;                             //!< Euclidean distance
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/frame-collision.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_TRANSLATION_HPP_
