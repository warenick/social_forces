import sys
import os
sys.path.append(os.path.abspath('../social_forces'))
from SFM import SFM
from Param import Param
from WorkerThread import WorkerThread, WorkerPThread, WorkerThread2,WorkerThread3, func_thread
import multiprocessing
import torch
from threading import Thread
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import Queue
# import queue
import time
from itertools import product

class SFM_AI():
    def __init__(self,param = None,device='cpu'):
        self.param = param
        if self.param is None:
            self.param = Param(device)
        self.sfm = SFM(self.param, device)

    def chunkIt(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def calc_speed_vect(self,state,goal,future_horizon):
        velocity = torch.abs((goal-state)/future_horizon)
        return velocity.norm(dim=1)

    def get_sfm_predictions(self, agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal, neighb_avail = None, future_horizon=12, num_threads=0,calc_speeds = False):
        # agent_state shape: bs, 2 torch.tensor
        # neighb_state shape  list of len bs, each elem is tensor with shape (n,2)
        # agent_vel: bs, 2 torch.tensor
        # neighb_vel:  list of len bs, each elem is tensor with shape (n,2)
        # agent_goal: bs, 2 torch.tensor
        # neighb_goal list of len bs, each elem is tensor with shape (n,2)
        # return tensor(bs, future_horizon, 2),   tensor(bs, future_horizon,  4) (edited)
        assert agent_state.ndim == 2
        assert agent_vel.ndim == 2
        assert neighb_state.ndim == 3
        assert neighb_vel.ndim == 3
        assert agent_goal.ndim == 2
        assert neighb_goal.ndim == 3
        dev = agent_state.device
        agent_f_state = torch.cat((agent_state, agent_vel), dim=1)
        neighb_f_state = torch.cat((neighb_state, neighb_vel), dim=2)
        state = torch.cat((agent_f_state.unsqueeze(1), neighb_f_state), dim=1)
        goals = torch.cat((agent_goal.unsqueeze(1), neighb_goal), dim=1)
        stacked_state = []
        stacked_forces = []
        batch_size = agent_state.shape[0]
        if neighb_avail is not None:
            neighb_avail = torch.cat((torch.ones(batch_size,1,device=dev,dtype=torch.bool),neighb_avail),dim=1)
        # process in main thread
        if num_threads == 0:
            for chunk in range(batch_size):
                step_stacked_state = []
                step_stacked_forces = []
                if neighb_avail is not None:
                    # TODO: add avail
                    local_state = torch.masked_select(state[chunk].clone(),neighb_avail[chunk],)
                    local_goals = torch.masked_select(goals[chunk],neighb_avail[chunk])
                else:
                    local_state = state[chunk].clone()
                    local_goals = goals[chunk]
                speed_vect = None
                if calc_speeds:
                    speed_vect = self.calc_speed_vect(local_state[:,:2],local_goals,future_horizon)
                    speed_vect*=self.param.socForceRobotPerson["k_calc_speed"]# 1.3
                    speed_vect/=self.sfm.param.DT
                    # print(speed_vect)
                for step in range(future_horizon):
                    rep_force, attr_force = self.sfm.calc_forces(local_state, local_goals,speed_vect)
                    F = rep_force+attr_force
                    local_state = self.sfm.pose_propagation(F, local_state.clone(),speed_vect)
                    local_forces = torch.cat((rep_force[0],attr_force[0]),dim=0)
                    step_stacked_forces.append(local_forces.tolist())
                    step_stacked_state.append(local_state[0].tolist())
                stacked_state.append(step_stacked_state)
                stacked_forces.append(step_stacked_forces)
            forces = torch.tensor(stacked_forces)
            poses = torch.tensor(stacked_state)
            return poses, forces
        # process in threads
        in_queue = Queue()
        out_queue = Queue()
        chunks = self.chunkIt(range(batch_size), num_threads)
        num = 0
        chunk_data = []
        for chunk in chunks:
            chunk_data.append((state[chunk], goals[chunk], future_horizon, num, SFM(self.param,dev)))
            num+=1
        return_val = []
        with Pool() as pool:
            return_val.append(pool.starmap(func_thread, chunk_data))
        poses_ = []
        forces_ = []
        nums = []
        for chunk in range(num_threads):
            (pose,force,num) = return_val[0][chunk]
            poses_.append(pose)
            forces_.append(force)
            nums.append(num)
        # # sort out
        _,indices = torch.sort(torch.tensor(nums))
        sorted_poses_ = []
        sorted_forces_ = []
        for n in indices:
            sorted_poses_.append(poses_[n])
            sorted_forces_.append(forces_[n])
        poses = torch.cat(sorted_poses_,dim=0)
        forces = torch.cat(sorted_forces_,dim=0)
        return poses, forces


if __name__ == '__main__':
    future_horizon = 12
    bs = 1000
    neighb_num = 10
    dev = 'cpu'
    # if torch.cuda.is_available():
    #     dev = torch.device(0)
    sfm_ai = SFM_AI(device=dev)
    agent_state = torch.rand((bs, 2),device =dev)
    neighb_state = torch.rand((bs, neighb_num, 2),device =dev)
    agent_vel = torch.rand((bs, 2),device =dev)
    neighb_vel = torch.rand((bs, neighb_num, 2),device =dev)
    agent_goal = torch.rand((bs, 2),device =dev)
    neighb_goal = torch.rand((bs, neighb_num, 2),device =dev)
    neighb_avail = torch.ones(bs,neighb_num,device=dev,dtype=torch.bool)
    neighb_avail = None
    # w/o threading
    start = time.time()
    poses, forces = sfm_ai.get_sfm_predictions(
        agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal,neighb_avail,future_horizon=12, num_threads =0)
    print("working time "+str(time.time()-start))
    print("w/o threading ok")
    # w threading
    start = time.time()
    poses, forces = sfm_ai.get_sfm_predictions(
        agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal, neighb_avail, future_horizon=12, num_threads=6)
    print("working time "+str(time.time()-start))
    print("w threading ok")
    exit()

