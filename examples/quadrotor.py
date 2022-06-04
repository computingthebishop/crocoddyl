import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ
WITHDISPLAY = True
WITHPLOT = True

hector = example_robot_data.load('hector')
robot_model = hector.model

target_pos = np.array([2., 0., 1.])
target_quat = pinocchio.Quaternion(1., 0., 0., 0.)
wp1_pos = np.array([0., -3., 1.])
wp1_quat = pinocchio.Quaternion(1., 0., 0., 0.)
motor_time_ct = 0.01

rotors = 4
state = crocoddyl.StateMultibodyActuated(robot_model,rotors) # create state model from pinocchio model

d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5., -5.0
tau_f = np.array([[0., 0., 0., 0.], 
                  [0., 0., 0., 0.], 
                  [1., 1., 1., 1.], 
                  [0., d_cog, 0., -d_cog],
                  [-d_cog, 0., d_cog, 0.], 
                  [-cm / cf, cm / cf, -cm / cf, cm / cf]])
                  
actuation = crocoddyl.ActuationModelMultiCopterBaseFos(state, tau_f) # using custom actuator class

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
wp1CostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Costs
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
#weights = np.array(([0.1] * 3) + ([1000.] * 3) + ([0.01] * 2 * rotors) + ([1000.] * robot_model.nv) + ([0.01] * rotors))  # x y z rx ry rz (a+bi)theta1...4 vx vy vz vrx vry vrz w1...4
weights = np.array(([0.1] * 3) + 
                  ([1000.] * 3) + 
                  ([0.0] * rotors) + 
                  ([1000.] * robot_model.nv) + 
                  ([0.001] * rotors))  # x y z rx ry rz (a+bi)theta1...4 vx vy vz vrx vry vrz w1...4
                 # x y z rx ry rz   vx vy vz vrx vry vrz
xActivation = crocoddyl.ActivationModelWeightedQuad(weights)


uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)


wp1Residual = crocoddyl.ResidualModelFramePlacementAugmented(state, robot_model.getFrameId("base_link"),
                                                            pinocchio.SE3(wp1_quat.matrix(), wp1_pos), nu)
wp1TrackingCost = crocoddyl.CostModelResidual(state, wp1Residual)

goalTrackingResidual = crocoddyl.ResidualModelFramePlacementAugmented(state, robot_model.getFrameId("base_link"),
                                                            pinocchio.SE3(target_quat.matrix(), target_pos), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)

runningCostModel.addCost("xReg", xRegCost, 1e-6)
runningCostModel.addCost("uReg", uRegCost, 1e-5)
# runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
wp1CostModel.addCost("wp1",wp1TrackingCost, 3e-1)
terminalCostModel.addCost("goalPose", goalTrackingCost, 3.)

dt = 9e-3
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamicsActuated(state, actuation, runningCostModel, motor_time_ct), dt)
wp1_Model = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamicsActuated(state, actuation, wp1CostModel, motor_time_ct), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamicsActuated(state, actuation, terminalCostModel, motor_time_ct), dt)
# runningModel = crocoddyl.IntegratedActionModelRK4(
#     crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
# wp1_Model = crocoddyl.IntegratedActionModelRK4(
#     crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, wp1CostModel), dt)
# terminalModel = crocoddyl.IntegratedActionModelRK4(
#     crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), dt)

runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])

# Creating the shooting problem and the FDDP solver
T = 330
initial_state = np.concatenate([hector.q0, ([0, 1] * rotors), np.zeros(state.nv)])

models_arr = []
for i in range(T):
    if i == int(T/2):
        models_arr.append(wp1_Model)
    else:
        models_arr.append(runningModel)

problem = crocoddyl.ShootingProblem(initial_state, [runningModel]*T, terminalModel)
solver = crocoddyl.SolverFDDP(problem)
# solver = crocoddyl.SolverBoxFDDP(problem)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

cameraTF = [-0.03, 6.6, 3.3, -0.02, 0.56, 0.83, -0.03]
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
