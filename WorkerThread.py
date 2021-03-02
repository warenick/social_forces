import time
from multiprocessing import Process
from threading import Thread
import torch

class WorkerPThread(Process):
    def __init__(self,prediction_model,in_queue,out_queue):
        Process.__init__(self)
        self.pm = prediction_model
        self.in_queue = in_queue
        self.out_queue = out_queue


    def run(self):
        (state,goals,future_horizon,num) = self.in_queue.get()
        batch_size = state.shape[0]
        stacked_forces = []
        stacked_state = []
        for chunk in range(batch_size):
            step_stacked_forces = []
            step_stacked_state = []
            local_state = state[chunk].clone()
            local_goals = goals[chunk]
            for step in range(future_horizon):
                rep_force, attr_force = self.pm.calc_forces(local_state, local_goals)
                F = rep_force+attr_force
                local_state = self.pm.pose_propagation(F, local_state.clone())
                forces = torch.cat((rep_force[0],attr_force[0]),dim=0)
                step_stacked_forces.append(forces.tolist())
                step_stacked_state.append(local_state[0].tolist())
            stacked_forces.append(step_stacked_forces)
            stacked_state.append(step_stacked_state)
        # self.out_queue.put((stacked_state,stacked_forces,num))
        
        forces = torch.tensor(stacked_forces,device=goals.device)
        state = torch.tensor(stacked_state,device=goals.device)
        self.out_queue.put((state,forces,num))
        while(not self.out_queue.empty()):
            time.sleep(0.1) # dirty way
        # time.sleep(1) 


class WorkerThread(Thread):
    def __init__(self,prediction_model,in_queue,out_queue):
        Thread.__init__(self)
        self.pm = prediction_model
        self.in_queue = in_queue
        self.out_queue = out_queue


    def run(self):
        # start = time.time()
        while(not self.in_queue.empty()):
            # time.sleep(.01)
            step_stacked_forces = []
            step_stacked_state = []
            (state,goals,future_horizon,num) = self.in_queue.get()
            # (data,future_horizon,num)  = self.in_queue.get()
            # state = data[:,:4]
            # goals = data[:,4:]
            # if data is None:
            #     break
            # check num ped in param
            # local_state = state.clone()
            for step in range(future_horizon):
                rep_force, attr_force = self.pm.calc_forces(local_state, goals)
                F = rep_force+attr_force
                local_state = self.pm.pose_propagation(F, local_state.clone())
                forces = torch.cat((rep_force[0],attr_force[0]),dim=0)
                step_stacked_forces.append(forces.tolist())
                step_stacked_state.append(local_state[0].tolist())
            self.out_queue.put((step_stacked_state,step_stacked_forces,num))
        # print("time"+str(time.time()-start))


# 2222222222222222222222222222222222222
class WorkerThread2(Thread):
    def __init__(self,prediction_model,in_queue,out_queue):
        Thread.__init__(self)
        self.pm = prediction_model
        self.in_queue = in_queue
        self.out_queue = out_queue


    def run(self):
        (state,goals,future_horizon,num) = self.in_queue.get()
        batch_size = state.shape[0]
        stacked_forces = []
        stacked_state = []
        for chunk in range(batch_size):
            step_stacked_forces = []
            step_stacked_state = []
            local_state = state[chunk].clone()
            local_goals = goals[chunk]
            for step in range(future_horizon):
                rep_force, attr_force = self.pm.calc_forces(local_state, local_goals)
                F = rep_force+attr_force
                local_state = self.pm.pose_propagation(F, local_state.clone())
                forces = torch.cat((rep_force[0],attr_force[0]),dim=0)
                step_stacked_forces.append(forces.tolist())
                step_stacked_state.append(local_state[0].tolist())
            stacked_forces.append(step_stacked_forces)
            stacked_state.append(step_stacked_state)
        forces = torch.tensor(stacked_forces,device=goals.device)
        state = torch.tensor(stacked_state,device=goals.device)
        self.out_queue.put((state,forces,num))
