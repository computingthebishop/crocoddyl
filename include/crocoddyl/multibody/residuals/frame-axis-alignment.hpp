///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_PLACEMENT_HPP_

#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/skew.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame placement residual
 *
 * This residual function defines the tracking of a frame axis as \f$\mathbf{r}=\mathbf{n}^{*\top}
 * \mathbf{R}\mathbf{e}_{axis}\f$, where \f$\mathbf{n}^{*}, \mathbf{R}, \mathbf{e}\f$ are reference axis, the frame
 * rotation and, the axis of the frame we want to align with the reference axis, respectively. Note that the dimension
 * of the residual vector is 1. Furthermore, the Jacobians of the residual function are computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFrameAxisAlignmentTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFrameAxisAlignmentTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::MatrixXs MatrixXs;


  /**
   * @brief Initialize the frame placement residual model
   *
   * @param[in] state     State of the multibody system
   * @param[in] id        Reference frame id
   * @param[in] axisref   Reference axis
   * @param[in] direction 0, 1, 2 to select X, Y, Z axis from the frame to be aligned with axisref
   * @param[in] nu        Dimension of the control vector
   */
  ResidualModelFrameAxisAlignmentTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                     const Vector3s& axisref, const std::size_t direction, const std::size_t nu);

  virtual ~ResidualModelFrameAxisAlignmentTpl();

  /**
   * @brief Compute the frame placement residual
   *
   * @param[in] data  Frame placement residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the frame placement residual
   *
   * @param[in] data  Frame-placement residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame placement residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference frame placement
   */
  const Vector3s& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference frame placement
   */
  void set_reference(const Vector3s& reference);

  /**
   * @brief Print relevant information of the frame-placement residual
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
  pinocchio::FrameIndex id_;  //!< Reference frame id
  Vector3s axisref_;          //!< Reference axis
  std::size_t direction_;     //!< 0,1,2 for X Y Z
  Vector3s v_axis_frame_;
  Matrix3s skew_axis_frame_;
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ResidualDataFrameAxisAlignmentTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::MatrixX3s MatrixX3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;

  template <template <typename Scalar> class Model>
  ResidualDataFrameAxisAlignmentTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), dotJvec(1, 3), rJf(3, model->get_state()->get_nv()), fJf(6, model->get_state()->get_nv()) {
    r.setZero();

    dotJvec.setZero();
    ractJrot.setZero();
    rJf.setZero();
    fJf.setZero();

    rJf.setZero();
    fJf.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data

  Vector3s axis;

  MatrixX3s dotJvec;  //!< Jacoabian of the dot product w.r.t. input vector
  Matrix3s ractJrot;  //!< Jacobian of the rotation action w.r.t. rotation
  Matrix3xs rJf;      //!< Local Jacobian of the frame (rotation part)
  Matrix6xs fJf;      //!< Local Jacobian of the frame

  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/frame-axis-alignment.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_PLACEMENT_HPP_
