import sys
import os
sys.path.append(os.path.abspath('../social_forces'))
from SFM import SFM
from Param import Param
from WorkerThread import WorkerThread, WorkerPThread, WorkerThread2
import multiprocessing
import torch

from multiprocessing import Queue
# import queue
import time


class SFM_AI():
    def __init__(self,device='cpu'):
        self.param = Param(device)
        self.sfm = SFM(self.param, device)

    def chunkIt(self,a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def get_sfm_predictions(self, agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal, neighb_avail = None, future_horizon=12, num_threads=0):
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
                for step in range(future_horizon):
                    rep_force, attr_force = self.sfm.calc_forces(local_state, local_goals)
                    F = rep_force+attr_force
                    local_state = self.sfm.pose_propagation(F, local_state.clone())
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
        thread_pull = []
        chunks = self.chunkIt(range(batch_size), num_threads)
        num = 0
        for chunk in chunks:
            in_queue.put((state[chunk], goals[chunk], future_horizon, num))
            num+=1
        for n in range(num_threads):
            thread_pull.append(WorkerPThread(SFM(self.param,dev),in_queue,out_queue))
            thread_pull[n].start()
        # for thread in thread_pull:
        #     thread.join()
        poses_ = []
        forces_ = []
        nums = []

        for chunk in range(num_threads):# dirty way
            (pose,force,num) = out_queue.get()
            poses_.append(pose)
            forces_.append(force)
            nums.append(num)
        # # sort out
        _,indices = torch.sort(torch.tensor(nums))
        # poses_ = torch.tensor(poses_,device=dev)[indices] #suka expected sequence of length 167 at dim 1 (got 166)
        # forces_ = torch.tensor(forces_,device=dev)[indices]
        sorted_poses_ = []
        sorted_forces_ = []
        for n in indices:
            sorted_poses_.append(poses_[n])
            sorted_forces_.append(forces_[n])
        poses = torch.cat(sorted_poses_,dim=0)
        forces = torch.cat(sorted_forces_,dim=0)
        for thread in thread_pull:
            thread.join()
        return poses, forces
        #         for chunk in range(batch_size):
        #     in_queue.put((state[chunk],goals[chunk],future_horizon,chunk))
        #     # in_queue.put((torch.cat((state[chunk],goals[chunk]), dim=1),future_horizon,chunk))
        # # for _ in range(num_threads):
        # #     in_queue.put(None)
        # # chunks = self.chunkIt(range(batch_size), num_threads)
        # # n = 0
        # # for chunk in chunks:
        # #     in_queue.put((state[chunk], goals[chunk], future_horizon, chunk, n))
        # #     n+=1

        # # start = time.time()
        
        # for n in range(num_threads):
        #     thread_pull.append(WorkerThread(SFM(self.param,dev),in_queue,out_queue))
        #     # thread_pull.append(WorkerPThread(SFM(self.param,dev),in_queue,out_queue))
        #     thread_pull[n].start()
        # # print("working in time1 "+str(time.time()-start))
        # for thread in thread_pull:
        #     thread.join()
        # poses_ = []
        # forces_ = []
        # nums = []

        # for chunk in range(out_queue.qsize()): #TODO: please check it qsize
        #     (pose,force,num) = out_queue.get()
        #     poses_.append(pose)
        #     forces_.append(force)
        #     nums.append(num)
        # # sort out
        # _,indices = torch.sort(torch.tensor(nums))
        # poses = torch.tensor(poses_)[indices]
        # forces = torch.tensor(forces_)[indices]

        # return poses, forces



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

