import time
from threading import Thread
import torch

class WorkerThread(Thread):
    def __init__(self,prediction_model,in_queue,out_queue):
        Thread.__init__(self)
        self.pm = prediction_model
        self.in_queue = in_queue
        self.out_queue = out_queue


    def run(self):
        while(not self.in_queue.empty()):
            # time.sleep(.001)
            step_stacked_forces = []
            step_stacked_state = []
            (state,goals,future_horizon,num) = self.in_queue.get()
            # check num ped in param
            for step in range(future_horizon):
                rep_force, attr_force = self.pm.calc_forces(state, goals)
                F = rep_force+attr_force
                state = self.pm.pose_propagation(F, state.clone())
                forces = torch.cat((rep_force[0],attr_force[0]),dim=0)
                step_stacked_forces.append(forces.tolist())
                step_stacked_state.append(state[0].tolist())
            self.out_queue.put((step_stacked_state,step_stacked_forces,num))