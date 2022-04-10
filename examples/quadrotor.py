import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHDISPLAY = True
WITHPLOT = False
ACTUATOR = True
#ACTUATOR = False

hector = example_robot_data.load('hector')
robot_model = hector.model

target_pos = np.array([1., 0., 1.])
target_quat = pinocchio.Quaternion(1., 0., 0., 0.)

if (ACTUATOR):
    rotors = 0
    state = crocoddyl.StateMultibodyActuated(robot_model,rotors) # create state model from pinocchio model
else:
    state = crocoddyl.StateMultibody(robot_model) # create state model from pinocchio model

d_cog, cf, cm = 0.1525, 6.6e-5, 1e-6
tau_f = np.array([[0., 0., 0., 0.], 
                  [0., 0., 0., 0.], 
                  [1., 1., 1., 1.], 
                  [0., d_cog, 0., -d_cog],
                  [-d_cog, 0., d_cog, 0.], 
                  [-cm / cf, cm / cf, -cm / cf, cm / cf]])
                  
#actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)
actuation = crocoddyl.ActuationModelMultiCopterBaseFos(state, tau_f) # using custom actuator class

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Costs
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
if (ACTUATOR):  
    #weights = np.array(([0.1] * 3) + ([1000.] * 3) + ([0.01] * 2 * rotors) + ([1000.] * robot_model.nv) + ([0.01] * rotors))  # x y z rx ry rz (a+bi)theta1...4 vx vy vz vrx vry vrz w1...4
    weights = np.array(([0.1] * 3) + ([1000.] * 3) + ([0.01] * rotors) + ([1000.] * robot_model.nv) + ([0.01] * rotors))  # x y z rx ry rz (a+bi)theta1...4 vx vy vz vrx vry vrz w1...4
else:
    weights = np.array([0.1] * 3 + [1000.] * 3 + [1000.] * robot_model.nv)                   # x y z rx ry rz   vx vy vz vrx vry vrz
xActivation = crocoddyl.ActivationModelWeightedQuad(weights)


uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("base_link"),
                                                             pinocchio.SE3(target_quat.matrix(), target_pos), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
runningCostModel.addCost("xReg", xRegCost, 1e-6)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
terminalCostModel.addCost("goalPose", goalTrackingCost, 3.)

dt = 3e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), dt)

# Creating the shooting problem and the FDDP solver
T = 33
if (ACTUATOR):  
    problem = crocoddyl.ShootingProblem(np.concatenate([hector.q0, np.zeros(state.nv), np.zeros(rotors)]), [runningModel] * T, terminalModel)
else:
    problem = crocoddyl.ShootingProblem(np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel)
solver = crocoddyl.SolverFDDP(problem)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

cameraTF = [-0.03, 4.4, 2.3, -0.02, 0.56, 0.83, -0.03]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the FDDP solver
solver.solve()

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.stops, log.grads, log.steps, figIndex=2)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(hector)
    hector.viewer.gui.addXYZaxis('world/wp', [1., 0., 0., 1.], .03, 0.5)
    hector.viewer.gui.applyConfiguration(
        'world/wp',
        target_pos.tolist() + [target_quat[0], target_quat[1], target_quat[2], target_quat[3]])

    display.displayFromSolver(solver)
