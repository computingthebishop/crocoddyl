///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, IRI; CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_AL_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_AL_DDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

class SolverALDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverALDDP(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverALDDP();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t& maxiter = 100,
                     const bool& is_feasible = false, const double& regInit = 1e-9);

  virtual double calcDiff();

  virtual void backwardPass();
  virtual void forwardPass(const double& stepLength);

  virtual void acceptableViolation();

  virtual void allocateData();

 protected:
  std::size_t maxiter_outer_;  //!< Maximum number of dual updates
  std::size_t outer_iter_;     //!< Counter of dual updates
  std::size_t inner_iter_;     //!< Counter of inner iterations
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
