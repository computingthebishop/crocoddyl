///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, IRI (CSIC - UPC)
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FULL_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FULL_HPP_

#include <stdexcept>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelQuadFullTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelQuadFullTpl(const VectorXs& w, const MatrixXs& W) : Base(w.size()), w_(w), W_(W){};
  virtual ~ActivationModelQuadFullTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    data->a_value = (w_.transpose() * r + (Scalar(0.5) * r.transpose() * W_ * r))[0];
  };

  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }

    data->Ar = w_ + W_ * r;
    data->Arr = W_;
  };

  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    boost::shared_ptr<ActivationDataAbstract> data =
        boost::allocate_shared<ActivationDataAbstract>(Eigen::aligned_allocator<ActivationDataAbstract>(), this);
    data->Arr = W_;
    return data;
  };

  const VectorXs& get_linear_weights() const { return w_; };
  const MatrixXs& get_quadratic_weights() const { return W_; };

  void set_linear_weights(const VectorXs& weights) {
    if (weights.size() != w_.size()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " + std::to_string(w_.size()) + ")");
    }
    w_ = weights;
  };

  void set_quadratic_weights(const MatrixXs& weights) {
    if (weights.rows() != W_.rows() && weights.cols() != W_.cols()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " + std::to_string(W_.rows()) + " x " +
                          std::to_string(W_.cols()) + ")");
    }
    W_ = weights;
  };

  /**
   * @brief Print relevant information of the quadratic model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const { os << "ActivationModelQuadFull {nr=" << nr_ << "}"; }

 protected:
  using Base::nr_;

 private:
  VectorXs w_;
  MatrixXs W_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_HPP_
