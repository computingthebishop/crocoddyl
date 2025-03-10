///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeCore() {
  exposeDataCollector();
  exposeStateAbstract();
  exposeControlParametrizationAbstract();
  exposeActuationAbstract();
  exposeActionAbstract();
  exposeIntegratedActionAbstract();
  exposeDifferentialActionAbstract();
  exposeResidualAbstract();
  exposeActivationAbstract();
  exposeSquashingAbstract();
  exposeSquashingSmoothSat();
  exposeActuationSquashing();
  exposeDataCollectorActuation();
  exposeIntegratedActionEuler();
  exposeIntegratedActionRK();
  exposeIntegratedActionRK4();
  exposeCostAbstract();
  exposeResidualControl();
  exposeCostSum();
  exposeCostResidual();
  exposeCostControl();
  exposeActionNumDiff();
  exposeDifferentialActionNumDiff();
  exposeActivationNumDiff();
  exposeStateNumDiff();
  exposeShootingProblem();
  exposeSolverAbstract();
  exposeStateEuclidean();
  exposeControlParametrizationPolyZero();
  exposeControlParametrizationPolyOne();
  exposeControlParametrizationPolyTwoRK();
  exposeActionUnicycle();
  exposeActionLQR();
  exposeDifferentialActionLQR();
  exposeActivationQuad();
  exposeActivationQuadFull();
  exposeActivationQuadFlatExp();
  exposeActivationQuadFlatLog();
  exposeActivationWeightedQuad();
  exposeActivationQuadraticBarrier();
  exposeActivationWeightedQuadraticBarrier();
  exposeActivationSmooth1Norm();
  exposeActivationSmooth2Norm();
  exposeActivation2NormBarrier();
  exposeSolverKKT();
  exposeSolverDDP();
  exposeSolverFDDP();
  exposeSolverBoxQP();
  exposeSolverBoxDDP();
  exposeSolverBoxFDDP();
  exposeSolverIpopt();
  exposeCallbacks();
  exposeStopWatch();
}

}  // namespace python
}  // namespace crocoddyl
