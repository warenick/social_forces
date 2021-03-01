from SFM import SFM
from Param import Param
from WorkerThread import WorkerThread
import torch
import queue
from threading import Thread
import time


class SFM_AI():
    def __init__(self):
        self.param = Param()
        self.sfm = SFM(self.param)

    def get_sfm_predictions(self, agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal, future_horizon=12, num_threads=0):
        # agent_state shape: bs, 2 torch.tensor
        # neighb_state shape  list of len bs, each elem is tensor with shape (n,2)
        # agent_vel: bs, 2 torch.tensor
        # neighb_vel:  list of len bs, each elem is tensor with shape (n,2)
        # agent_goal: bs, 2 torch.tensor
        # neighb_goal list of len bs, each elem is tensor with shape (n,2)
        # return tensor(bs, future_horizon, 2),   tensor(bs, future_horizon,  4) (edited)
        agent_f_state = torch.cat((agent_state, agent_vel), dim=1)
        neighb_f_state = torch.cat((neighb_state, neighb_vel), dim=2)
        state = torch.cat((agent_f_state.unsqueeze(1), neighb_f_state), dim=1)
        goals = torch.cat((agent_goal.unsqueeze(1), neighb_goal), dim=1)
        stacked_state = []
        stacked_forces = []
        batch_size = agent_state.shape[0]
        # process in main thread
        if num_threads == 0:
            for chunk in range(batch_size):
                step_stacked_state = []
                step_stacked_forces = []
                local_state = state[chunk].clone()
                # check num peds in param
                for step in range(future_horizon):
                    rep_force, attr_force = self.sfm.calc_forces(local_state, goals[chunk])
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
        in_queue = queue.Queue()
        out_queue = queue.Queue()
        thread_pull = []
        for chunk in range(batch_size):
            in_queue.put((state[chunk],goals[chunk],future_horizon,chunk))
        for n in range(num_threads):
            thread_pull.append(WorkerThread(SFM(self.param),in_queue,out_queue))
            thread_pull[n].start()
        for thread in thread_pull:
            thread.join()
        poses_ = []
        forces_ = []
        nums = []
        for chunk in range(out_queue.qsize()): #TODO: please check it qsize
            (pose,force,num) = out_queue.get()
            poses_.append(pose)
            forces_.append(force)
            nums.append(num)
        # sort out
        _,indices = torch.sort(torch.tensor(nums))
        poses = torch.tensor(poses_)[indices]
        forces = torch.tensor(forces_)[indices]
        return poses, forces
        



if __name__ == '__main__':
    sfm_ai = SFM_AI()
    future_horizon = 12
    bs = 1000
    neighb_num = 10
    agent_state = torch.rand((bs, 2))
    neighb_state = torch.rand((bs, neighb_num, 2))
    agent_vel = torch.rand((bs, 2))
    neighb_vel = torch.rand((bs, neighb_num, 2))
    agent_goal = torch.rand((bs, 2))
    neighb_goal = torch.rand((bs, neighb_num, 2))
    # w/o threading
    start = time.time()
    poses, forces = sfm_ai.get_sfm_predictions(
        agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal, future_horizon, num_threads =0)
    print("working time "+str(time.time()-start))
    print("w/o threading ok")
    # w threading
    start = time.time()
    poses, forces = sfm_ai.get_sfm_predictions(
        agent_state, neighb_state, agent_vel, neighb_vel, agent_goal, neighb_goal, future_horizon, num_threads=6)
    print("working time "+str(time.time()-start))
    print("w threading ok")
    exit()

