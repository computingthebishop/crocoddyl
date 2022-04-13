///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/states/multibody_actuated.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl {
// PinocchioModel -> typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
// StateAbstractTpl<Scalar> Base; -> constructor Dimension of state configuration tuple
//                                               Dimension of state tangent vector
//
// nq -> Dimension of the configuration vector representation. - 3 POSITION 4 QUATERNION  (3+4=7)
// nv -> Dimension of the velocity vector space.               - 3 POSITION 3 ORIENTATION (3+3=6)
template <typename Scalar>
StateMultibodyActuatedTpl<Scalar>::StateMultibodyActuatedTpl(boost::shared_ptr<PinocchioModel> model, std::size_t nr)
    : pinocchio_(model), Base(model), x0_(VectorXs::Zero(model->nq + (2*nr) + model->nv + nr)), nr_(nr) {
  std::cout<< "[StateMultibodyActuatedTpl]" << std::endl;
  x0_.head(model->nq) = pinocchio::neutral(*pinocchio_.get());
  for (std::size_t i = 0; i < nr_; i++)
  {
    x0_(model->nq+(2*i)) = 1;
  }

  nq_ = model->nq + 2*nr_;
  nv_ = model->nv + nr_;
  nx_ = nq_ + nv_;
  ndx_ = 2*nv_;
}

template <typename Scalar>
StateMultibodyActuatedTpl<Scalar>::StateMultibodyActuatedTpl() : Base(), x0_(VectorXs::Zero(0)) {}

template <typename Scalar>
StateMultibodyActuatedTpl<Scalar>::~StateMultibodyActuatedTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateMultibodyActuatedTpl<Scalar>::zero() const {
  return x0_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateMultibodyActuatedTpl<Scalar>::rand() const {
  VectorXs xrand = VectorXs::Random(nx_);
  xrand.head(nq_-(2*nr_)) = pinocchio::randomConfiguration(*pinocchio_.get());
  //TODO normalize the complex number representing random rotor pose
  for (std::size_t i = 0; i < nr_; i++)
  {
    xrand(pinocchio_->nq+(2*i)) = 1;
    xrand(pinocchio_->nq+(2*i)+1) = 0;
  }
  return xrand;
}

template <typename Scalar>
void StateMultibodyActuatedTpl<Scalar>::diff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                     Eigen::Ref<VectorXs> dxout) const {
  // std::cout<< "[StateMultibodyActuatedTpl::diff]" << std::endl;
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dxout.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }

  pinocchio::difference(*pinocchio_.get(), x0.head(nq_-(2*nr_)), x1.head(nq_-(2*nr_)), dxout.head(nv_-nr_));
  for (std::size_t i = 0; i < nr_; i++)
  {
    // Conjugate of x0
    Scalar a = x0(nq_-(2*nr_)+(i*2));
    Scalar b = -x0(nq_-(2*nr_)+(i*2)+1);
    // Manifold point x1
    Scalar c = x1(nq_-(2*nr_)+(i*2));
    Scalar d = x1(nq_-(2*nr_)+(i*2)+1);
    // Product between conj(x0)*x1
    Scalar ar = (a*c)-(b*d);
    Scalar br = (a*d)+(b*c);
    // Log map
    Scalar thetad = atan2(br,ar);
    dxout(nq_-(2*nr_)+i) = thetad;
  }
  dxout.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
}

template <typename Scalar>
void StateMultibodyActuatedTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                          Eigen::Ref<VectorXs> xout) const {
  //std::cout<< "[StateMultibodyActuatedTpl::integrate]" << std::endl;
  if (static_cast<std::size_t>(x.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(dx.size()) != ndx_) {
    throw_pretty("Invalid argument: "
                 << "dx has wrong dimension (it should be " + std::to_string(ndx_) + ")");
  }
  if (static_cast<std::size_t>(xout.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "xout has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }

  pinocchio::integrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), xout.head(nq_-(2*nr_)));
  for (std::size_t i = 0; i < nr_; i++)
  {
    // Manifold point x
    Scalar a = x(nq_-(2*nr_)+(i*2));
    Scalar b = x(nq_-(2*nr_)+(i*2)+1);
    // Exp map of tangent space vector
    Scalar theta = dx(nq_-nr_+i);
    Scalar c = cos(theta);
    Scalar d = sin(theta);
    // Product between x and exp map of dx
    Scalar as = (a*c)-(b*d);
    Scalar bs = (a*d)+(b*c);
    xout(nq_-(2*nr_)+(i*2)) = as;
    xout(nq_-(2*nr_)+(i*2)+1) = bs;
  }
  xout.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
}

template <typename Scalar>
void StateMultibodyActuatedTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1,
                                      Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                      const Jcomponent firstsecond) const {
  // std::cout<< "[StateMultibodyActuatedTpl::Jdiff]" << std::endl;
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  if (static_cast<std::size_t>(x1.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }

  if (firstsecond == first) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }

    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_-(2*nr_)), x1.head(nq_-(2*nr_)), Jfirst.topLeftCorner(nv_-nr_, nv_-nr_),
                           pinocchio::ARG0);
    Jfirst.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() = (Scalar)-1;      //wrt x0
    Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)-1;
    // std::cout<< "Jacobian first" << std::endl;
    // std::cout<< Jfirst << std::endl;
  } else if (firstsecond == second) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_-(2*nr_)), x1.head(nq_-(2*nr_)), Jsecond.topLeftCorner(nv_-nr_, nv_-nr_),
                           pinocchio::ARG1);
    Jsecond.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() = (Scalar)1;     //wrt x1
    Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
    // std::cout<< "Jacobian second" << std::endl;
    // std::cout<< Jsecond << std::endl;
  } else {  // computing both
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_-(2*nr_)), x1.head(nq_-(2*nr_)), Jfirst.topLeftCorner(nv_-nr_, nv_-nr_),
                           pinocchio::ARG0);
    pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_-(2*nr_)), x1.head(nq_-(2*nr_)), Jsecond.topLeftCorner(nv_-nr_, nv_-nr_),
                           pinocchio::ARG1);
    Jfirst.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() = (Scalar)-1;      //wrt x0
    Jsecond.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() = (Scalar)-1;     //wrt x1
    Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
    Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
    // std::cout<< "Jacobian both" << std::endl;
    // std::cout<< Jfirst << std::endl;
    // std::cout<< Jsecond << std::endl;
  }
}

template <typename Scalar>
void StateMultibodyActuatedTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx,
                                           Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                           const Jcomponent firstsecond, const AssignmentOp op) const {
  //std::cout<< "[StateMultibodyActuatedTpl::Jintegrate]" << std::endl;
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (firstsecond == first || firstsecond == both) {
    if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                          ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jfirst.topLeftCorner(nv_-nr_, nv_-nr_),
                              pinocchio::ARG0, pinocchio::SETTO);
        Jfirst.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() = (Scalar)1;
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jfirst.topLeftCorner(nv_-nr_, nv_-nr_),
                              pinocchio::ARG0, pinocchio::ADDTO);
        Jfirst.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() += (Scalar)1;
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jfirst.topLeftCorner(nv_-nr_, nv_-nr_),
                              pinocchio::ARG0, pinocchio::RMTO);
        Jfirst.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() -= (Scalar)1;
        Jfirst.bottomRightCorner(nv_, nv_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
  if (firstsecond == second || firstsecond == both) {
    if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_) {
      throw_pretty("Invalid argument: "
                   << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                          std::to_string(ndx_) + ")");
    }
    switch (op) {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jsecond.topLeftCorner(nv_-nr_, nv_-nr_),
                              pinocchio::ARG1, pinocchio::SETTO);
        Jsecond.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() = (Scalar)1;
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jsecond.topLeftCorner(nv_-nr_, nv_-nr_),
                              pinocchio::ARG1, pinocchio::ADDTO);
        Jsecond.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() += (Scalar)1;
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jsecond.topLeftCorner(nv_-nr_, nv_-nr_),
                              pinocchio::ARG1, pinocchio::RMTO);
        Jsecond.block(nq_-nr_,nq_-nr_,nr_,nr_).diagonal().array() -= (Scalar)1;
        Jsecond.bottomRightCorner(nv_, nv_).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
    }
  }
}

template <typename Scalar>
void StateMultibodyActuatedTpl<Scalar>::JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                                                    const Eigen::Ref<const VectorXs>& dx, Eigen::Ref<MatrixXs> Jin,
                                                    const Jcomponent firstsecond) const {
  //std::cout<< "[StateMultibodyActuatedTpl::JintegrateTransport]" << std::endl;
  //TODO: check this function
  assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));

  switch (firstsecond) {
    case first:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jin.topRows(nv_-nr_), pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_-(2*nr_)), dx.head(nv_-nr_), Jin.topRows(nv_-nr_), pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. both not supported for this operation.");
      break;
  }
}

template <typename Scalar>
const boost::shared_ptr<pinocchio::ModelTpl<Scalar> >& StateMultibodyActuatedTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
std::size_t StateMultibodyActuatedTpl<Scalar>::get_nrotors() const {
  return nr_;
}

}  // namespace crocoddyl
