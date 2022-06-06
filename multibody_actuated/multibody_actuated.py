import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import eigenpy

class Cmultibody_actuated(crocoddyl.StateAbstract):
    def __init__(self, model, nr):
        nx = model.nq + model.nv + nr   #Dimension of state configuration tuple
                                        #model.nq -> Dimension of the configuration vector representation.      - 3 POSITION 4 QUATERNION  (3+4=7)
                                        #model.nv -> Dimension of the velocity vector space.                    - 3 POSITION 3 ORIENTATION (3+3=6)
                                        #      nr -> Dimension of the rotor configuration vector representation - 4 SPEEDS
        ndx = (2 * model.nv) + nr                                #Dimension of state tangent vector
        self.nr = nr
        super(Cmultibody_actuated, self).__init__(nx, ndx)       #Base state object
        self.pinocchio_ = model                                  #pinocchio model
        self.x0_ = np.zeros(model.nq + model.nv + nr)            #zero state vector
        self.x0_[:model.nq] = pinocchio.neutral(self.pinocchio_) #setting neutral configuration space
        
        self.nv = int((self.ndx-self.nr) / 2)                    #updating nv with correct dimension
        self.nq = int(self.nx - self.nr - self.nv)               #updating nq with correct dimension

    def zero(self):
        return self.x0_

    def rand(self):
        xrand = np.random.rand(self.nx)
        xrand[:self.nq] = pinocchio.randomConfiguration(self.pinocchio_)
        return xrand

    def diff(self, x0, x1, dxout):
        pinocchio.difference(self.pinocchio_, x0[:self.nq], x1[:self.nq], dxout[:self.nv])

    def integrate(self, x, dx, xout):
        pass

class CActuationModelMultiCopterBaseFos(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, tau_f):
        super(CActuationModelMultiCopterBaseFos, self).__init__(state, state.nv - 6 + len(tau_f[0]))       #Base actuation object
        self.n_rotors_ = len(tau_f[0])
        self.tau_f_ = np.zeros((state.nv, self.nu))
        self.tau_f_ = tau_f

    def calc(self, data, u):
        data.tau.noalias = self.tau_f_*u


if __name__ == "__main__":
    WITHDISPLAY = True
    WITHPLOT = False
    
    ACTUATOR = True
    #ACTUATOR = False

    hector = example_robot_data.load('hector')
    robot_model = hector.model

    target_pos = np.array([1., 0., 1.])
    target_quat = pinocchio.Quaternion(1., 0., 0., 0.)

    if (ACTUATOR):  
        rotors = 4
        state = Cmultibody_actuated(robot_model,rotors) 
    else:
        state = crocoddyl.StateMultibody(robot_model) # create state model from pinocchio model       

    d_cog, cf, cm = 0.1525, 6.6e-5, 1e-6
    tau_f = np.array([[      0.,      0.,       0.,      0.], 
                      [      0.,      0.,       0.,      0.], 
                      [      1.,      1.,       1.,      1.], 
                      [      0.,   d_cog,       0.,  -d_cog],
                      [  -d_cog,      0.,    d_cog,      0.], 
                      [-cm / cf, cm / cf, -cm / cf, cm / cf]])

    if (ACTUATOR):                
        actuation = CActuationModelMultiCopterBaseFos(state, tau_f) # using custom actuator class
    else:
        actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    # Costs
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu) #This residual function defines the state tracking 
                                                                      #State of the multibody system
                                                                      #Reference state
                                                                      #Dimension of the control vector
    if (ACTUATOR):                
        weights = np.array([0.1] * 3 + [1000.] * 3 + [1000.] * robot_model.nv + [0.01] * rotors)
    else:
        weights = np.array([0.1] * 3 + [1000.] * 3 + [1000.] * robot_model.nv)
    xActivation = crocoddyl.ActivationModelWeightedQuad(weights)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
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