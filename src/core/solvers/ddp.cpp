///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

SolverDDP::SolverDDP(boost::shared_ptr<ShootingProblem> problem)
    : SolverAbstract(problem),
      reg_incfactor_(10.),
      reg_decfactor_(10.),
      reg_min_(1e-9),
      reg_max_(1e9),
      cost_prev_(0.),
      cost_try_(0.),
      th_grad_(1e-12),
      th_stepdec_(0.5),
      th_stepinc_(0.01) {
  stopping_criteria_ = std::bind(&SolverDDP::stoppingCriteriaQuNorm, this);
  stopping_test_ = std::bind(&SolverDDP::stoppingTestFeasible, this);
  allocateData();

  const std::size_t n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
  if (th_stepinc_ < alphas_[n_alphas - 1]) {
    th_stepinc_ = alphas_[n_alphas - 1];
    std::cerr << "Warning: th_stepinc has higher value than lowest alpha value, set to "
              << std::to_string(alphas_[n_alphas - 1]) << std::endl;
  }
}

SolverDDP::~SolverDDP() {}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t maxiter, const bool is_feasible, const double reginit) {
  START_PROFILER("SolverDDP::solve");
  if (problem_->is_updated()) {
    resizeData();
  }
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    xreg_ = reg_min_;
    ureg_ = reg_min_;
  } else {
    xreg_ = reginit;
    ureg_ = reginit;
  }
  was_feasible_ = false;

  bool recalcDiff = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    while (true) {
      try {
        computeDirection(recalcDiff);
      } catch (std::exception& e) {
        recalcDiff = false;
        increaseRegularization();
        if (xreg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    expectedImprovement();

    // We need to recalculate the derivatives when the step length passes
    recalcDiff = false;
    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

      if (dVexp_ >= 0) {  // descend direction
        if (abs(d_[0]) < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, true);
          cost_prev_ = cost_;
          cost_ = cost_try_;
          recalcDiff = true;
          break;
        }
      }
    }

    if (steplength_ > th_stepdec_) {
      decreaseRegularization();
    }
    if (steplength_ <= th_stepinc_) {
      increaseRegularization();
      if (xreg_ == reg_max_) {
        STOP_PROFILER("SolverDDP::solve");
        return false;
      }
    }
    stoppingCriteria();

    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      CallbackAbstract& callback = *callbacks_[c];
      callback(*this);
    }

    if (stoppingTest()) {
      return true;
    }
  }
  STOP_PROFILER("SolverDDP::solve");
  return false;
}

void SolverDDP::computeDirection(const bool recalcDiff) {
  START_PROFILER("SolverDDP::computeDirection");
  if (recalcDiff) {
    calcDiff();
  }
  backwardPass();
  STOP_PROFILER("SolverDDP::computeDirection");
}

double SolverDDP::tryStep(const double steplength) {
  START_PROFILER("SolverDDP::tryStep");
  forwardPass(steplength);
  STOP_PROFILER("SolverDDP::tryStep");
  return cost_ - cost_try_;
}

double SolverDDP::stoppingCriteriaQuNorm() {
  stop_ = 0.;
  const std::size_t T = this->problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();

  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nu = models[t]->get_nu();
    if (nu != 0) {
      stop_ += Qu_[t].squaredNorm();
    }
  }
  return stop_;
}

double SolverDDP::stoppingCriteriaCostReduction() {
  stop_ = 0.;
  stop_ = std::abs(cost_ - cost_prev_) / cost_;
  return stop_;
}

double SolverDDP::stoppingCriteria() { return stopping_criteria_(); }

bool SolverDDP::stoppingTestFeasible() { return (was_feasible_ && stop_ < th_stop_); }

bool SolverDDP::stoppingTest() { return stopping_test_(); }

const Eigen::Vector2d& SolverDDP::expectedImprovement() {
  d_.fill(0);
  const std::size_t T = this->problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nu = models[t]->get_nu();
    if (nu != 0) {
      d_[0] += Qu_[t].dot(k_[t]);
      d_[1] -= k_[t].dot(Quuk_[t]);
    }
  }
  return d_;
}

void SolverDDP::resizeData() {
  START_PROFILER("SolverDDP::resizeData");
  SolverAbstract::resizeData();

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Qxu_[t].conservativeResize(ndx, nu);
    Quu_[t].conservativeResize(nu, nu);
    Qu_[t].conservativeResize(nu);
    K_[t].conservativeResize(nu, ndx);
    k_[t].conservativeResize(nu);
    us_try_[t].conservativeResize(nu);
    FuTVxx_p_[t].conservativeResize(nu, ndx);
    Quuk_[t].conservativeResize(nu);
  }
  STOP_PROFILER("SolverDDP::resizeData");
}

double SolverDDP::calcDiff() {
  START_PROFILER("SolverDDP::calcDiff");
  if (iter_ == 0) {
    problem_->calc(xs_, us_);
  }
  cost_ = problem_->calcDiff(xs_, us_);

  ffeas_ = computeDynamicFeasibility();
  STOP_PROFILER("SolverDDP::calcDiff");
  return cost_;
}

void SolverDDP::backwardPass() {
  START_PROFILER("SolverDDP::backwardPass");
  const boost::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx;

  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal().array() += xreg_;
  }

  if (!is_feasible_) {
    Vx_.back().noalias() += Vxx_.back() * fs_.back();
  }
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];
    const std::size_t nu = m->get_nu();

    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    START_PROFILER("SolverDDP::Qx");
    Qx_[t] = d->Lx;
    Qx_[t].noalias() += d->Fx.transpose() * Vx_p;
    STOP_PROFILER("SolverDDP::Qx");
    START_PROFILER("SolverDDP::Qxx");
    Qxx_[t] = d->Lxx;
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    STOP_PROFILER("SolverDDP::Qxx");
    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      START_PROFILER("SolverDDP::Qu");
      Qu_[t] = d->Lu;
      Qu_[t].noalias() += d->Fu.transpose() * Vx_p;
      STOP_PROFILER("SolverDDP::Qu");
      START_PROFILER("SolverDDP::Quu");
      Quu_[t] = d->Luu;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      STOP_PROFILER("SolverDDP::Quu");
      START_PROFILER("SolverDDP::Qxu");
      Qxu_[t] = d->Lxu;
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      STOP_PROFILER("SolverDDP::Qxu");
      if (!std::isnan(ureg_)) {
        Quu_[t].diagonal().array() += ureg_;
      }
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      START_PROFILER("SolverDDP::Vxx");
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
      STOP_PROFILER("SolverDDP::Vxx");
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;

    if (!std::isnan(xreg_)) {
      Vxx_[t].diagonal().array() += xreg_;
    }

    // Compute and store the Vx gradient at end of the interval (rollout state)
    if (!is_feasible_) {
      Vx_[t].noalias() += Vxx_[t] * fs_[t];
    }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverDDP::backwardPass");
}

void SolverDDP::forwardPass(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  START_PROFILER("SolverDDP::forwardPass");
  cost_try_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];

    m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
    if (m->get_nu() != 0) {
      us_try_[t].noalias() = us_[t];
      us_try_[t].noalias() -= k_[t] * steplength;
      us_try_[t].noalias() -= K_[t] * dx_[t];
      m->calc(d, xs_try_[t], us_try_[t]);
    } else {
      m->calc(d, xs_try_[t]);
    }
    xs_try_[t + 1] = d->xnext;
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverDDP::forwardPass");
      throw_pretty("forward_error");
    }
    if (raiseIfNaN(xs_try_[t + 1].lpNorm<Eigen::Infinity>())) {
      STOP_PROFILER("SolverDDP::forwardPass");
      throw_pretty("forward_error");
    }
  }

  const boost::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;

  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverDDP::forwardPass");
    throw_pretty("forward_error");
  }
  STOP_PROFILER("SolverDDP::forwardPass");
}

void SolverDDP::computeGains(const std::size_t t) {
  START_PROFILER("SolverDDP::computeGains");
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    START_PROFILER("SolverDDP::Quu_inv");
    Quu_llt_[t].compute(Quu_[t]);
    STOP_PROFILER("SolverDDP::Quu_inv");
    const Eigen::ComputationInfo& info = Quu_llt_[t].info();
    if (info != Eigen::Success) {
      STOP_PROFILER("SolverDDP::computeGains");
      throw_pretty("backward_error");
    }
    K_[t] = Qxu_[t].transpose();

    START_PROFILER("SolverDDP::Quu_inv_Qux");
    Quu_llt_[t].solveInPlace(K_[t]);
    STOP_PROFILER("SolverDDP::Quu_inv_Qux");
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
  }
  STOP_PROFILER("SolverDDP::computeGains");
}

void SolverDDP::increaseRegularization() {
  xreg_ *= reg_incfactor_;
  if (xreg_ > reg_max_) {
    xreg_ = reg_max_;
  }
  ureg_ = xreg_;
}

void SolverDDP::decreaseRegularization() {
  xreg_ /= reg_decfactor_;
  if (xreg_ < reg_min_) {
    xreg_ = reg_min_;
  }
  ureg_ = xreg_;
}

void SolverDDP::allocateData() {
  const std::size_t T = problem_->get_T();
  Vxx_.resize(T + 1);
  Vx_.resize(T + 1);
  Qxx_.resize(T);
  Qxu_.resize(T);
  Quu_.resize(T);
  Qx_.resize(T);
  Qu_.resize(T);
  K_.resize(T);
  k_.resize(T);

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T);

  FuTVxx_p_.resize(T);
  Quu_llt_.resize(T);
  Quuk_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Vxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Vx_[t] = Eigen::VectorXd::Zero(ndx);
    Qxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Qxu_[t] = Eigen::MatrixXd::Zero(ndx, nu);
    Quu_[t] = Eigen::MatrixXd::Zero(nu, nu);
    Qx_[t] = Eigen::VectorXd::Zero(ndx);
    Qu_[t] = Eigen::VectorXd::Zero(nu);
    K_[t] = MatrixXdRowMajor::Zero(nu, ndx);
    k_[t] = Eigen::VectorXd::Zero(nu);

    if (t == 0) {
      xs_try_[t] = problem_->get_x0();
    } else {
      xs_try_[t] = model->get_state()->zero();
    }
    us_try_[t] = Eigen::VectorXd::Zero(nu);
    dx_[t] = Eigen::VectorXd::Zero(ndx);

    FuTVxx_p_[t] = MatrixXdRowMajor::Zero(nu, ndx);
    Quu_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nu);
    Quuk_[t] = Eigen::VectorXd(nu);
  }
  Vxx_.back() = Eigen::MatrixXd::Zero(ndx, ndx);
  Vxx_tmp_ = Eigen::MatrixXd::Zero(ndx, ndx);
  Vx_.back() = Eigen::VectorXd::Zero(ndx);
  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();

  FxTVxx_p_ = MatrixXdRowMajor::Zero(ndx, ndx);
  fTVxx_p_ = Eigen::VectorXd::Zero(ndx);
}

double SolverDDP::get_reg_incfactor() const { return reg_incfactor_; }

double SolverDDP::get_reg_decfactor() const { return reg_decfactor_; }

double SolverDDP::get_regfactor() const { return reg_incfactor_; }

double SolverDDP::get_reg_min() const { return reg_min_; }

double SolverDDP::get_regmin() const { return reg_min_; }

double SolverDDP::get_reg_max() const { return reg_max_; }

double SolverDDP::get_regmax() const { return reg_max_; }

const std::vector<double>& SolverDDP::get_alphas() const { return alphas_; }

double SolverDDP::get_th_stepdec() const { return th_stepdec_; }

double SolverDDP::get_th_stepinc() const { return th_stepinc_; }

double SolverDDP::get_th_grad() const { return th_grad_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Vxx() const { return Vxx_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_Vx() const { return Vx_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Qxx() const { return Qxx_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Qxu() const { return Qxu_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Quu() const { return Quu_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_Qx() const { return Qx_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_Qu() const { return Qu_; }

const std::vector<typename MathBaseTpl<double>::MatrixXsRowMajor>& SolverDDP::get_K() const { return K_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_k() const { return k_; }

void SolverDDP::set_reg_incfactor(const double regfactor) {
  if (regfactor <= 1.) {
    throw_pretty("Invalid argument: "
                 << "reg_incfactor value is higher than 1.");
  }
  reg_incfactor_ = regfactor;
}

void SolverDDP::set_reg_decfactor(const double regfactor) {
  if (regfactor <= 1.) {
    throw_pretty("Invalid argument: "
                 << "reg_decfactor value is higher than 1.");
  }
  reg_decfactor_ = regfactor;
}

void SolverDDP::set_regfactor(const double regfactor) {
  if (regfactor <= 1.) {
    throw_pretty("Invalid argument: "
                 << "regfactor value is higher than 1.");
  }
  set_reg_incfactor(regfactor);
  set_reg_decfactor(regfactor);
}

void SolverDDP::set_reg_min(const double regmin) {
  if (0. > regmin) {
    throw_pretty("Invalid argument: "
                 << "regmin value has to be positive.");
  }
  reg_min_ = regmin;
}

void SolverDDP::set_regmin(const double regmin) {
  if (0. > regmin) {
    throw_pretty("Invalid argument: "
                 << "regmin value has to be positive.");
  }
  reg_min_ = regmin;
}

void SolverDDP::set_reg_max(const double regmax) {
  if (0. > regmax) {
    throw_pretty("Invalid argument: "
                 << "regmax value has to be positive.");
  }
  reg_max_ = regmax;
}

void SolverDDP::set_regmax(const double regmax) {
  if (0. > regmax) {
    throw_pretty("Invalid argument: "
                 << "regmax value has to be positive.");
  }
  reg_max_ = regmax;
}

void SolverDDP::set_alphas(const std::vector<double>& alphas) {
  double prev_alpha = alphas[0];
  if (prev_alpha != 1.) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    double alpha = alphas[i];
    if (0. >= alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
}

void SolverDDP::set_th_stepdec(const double th_stepdec) {
  if (0. >= th_stepdec || th_stepdec > 1.) {
    throw_pretty("Invalid argument: "
                 << "th_stepdec value should between 0 and 1.");
  }
  th_stepdec_ = th_stepdec;
}

void SolverDDP::set_th_stepinc(const double th_stepinc) {
  if (0. >= th_stepinc || th_stepinc > 1.) {
    throw_pretty("Invalid argument: "
                 << "th_stepinc value should between 0 and 1.");
  }
  th_stepinc_ = th_stepinc;
}

void SolverDDP::set_th_grad(const double th_grad) {
  if (0. > th_grad) {
    throw_pretty("Invalid argument: "
                 << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

void SolverDDP::set_stoppingCriteria(SolverDDP::StoppingType stop_type) {
  switch (stop_type) {
    case SolverDDP::StopCriteriaQuNorm:
      stopping_criteria_ = std::bind(&SolverDDP::stoppingCriteriaQuNorm, this);
      break;
    case SolverDDP::StopCriteriaCostReduction:
      stopping_criteria_ = std::bind(&SolverDDP::stoppingCriteriaCostReduction, this);
      break;
    default:
      stopping_criteria_ = std::bind(&SolverDDP::stoppingCriteriaQuNorm, this);
      break;
  }
}

}  // namespace crocoddyl
