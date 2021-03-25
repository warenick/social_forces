import torch
import numpy as np
import math


class RepulsiveForces():
    def __init__(self, param,device='cpu'):
        self.param = param
        self.num_ped = None
        self.aux1 = None
        self.auxullary = None
        self.aux = None
        self.indexes = None
        self.uneven_indexes = None
        self.even_indexes = None
        self.device = device

    def change_num_of_ped(self, new_num):
        self.num_ped = new_num
        self.aux1 = None
        self.aux2 = None
        self.auxullary = None
        self.aux = None
        self.indexes = None
        self.uneven_indexes = None
        self.even_indexes = None
        self.generate_aux_matrices()

    def generate_aux_matrices(self):
        if self.aux1 is None:
            self.aux1 = torch.tensor(([1., 0.], [0., 1.]),device=self.device)
            for i in range(0, self.num_ped):
                temp = torch.tensor(([1., 0.], [0., 1.]),device=self.device)
                self.aux1 = torch.cat((self.aux1, temp), dim=1)
            '''
                e.g. : state   [[1,  0]
                                [2,  1]
                                [-1,-1]
                                [0, -1]]
                new state_concated:         
                               [[1,  0, 1,  0, 1,  0, 1,  0]
                                [2,  1, 2,  1, 2,  1, 2,  1]
                                [-1,-1, -1,-1, -1,-1, -1,-1]
                                [0, -1, 0, -1, 0, -1, 0, -1]]
            '''
        if self.auxullary is None:
            self.auxullary = torch.zeros(self.num_ped+1, (self.num_ped+1)*2,device=self.device)
            for i in range(self.num_ped+1):
                self.auxullary[i, 2*i] = 1.
                self.auxullary[i, 2*i+1] = 1.

                ''' used to calc x+y of each agent pose
                    auxullary  tensor([
                        [1., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 1.]])
                '''
        if self.aux is None:
            self.aux = self.auxullary.t()
            
        if self.indexes is None:
            self.indexes = np.linspace(
                0., (self.num_ped+1)*2, ((self.num_ped+1)*2+1))
            self.uneven_indexes = self.indexes[1:-1:2]
            self.even_indexes = self.indexes[0:-1:2]

        if self.aux2 is None:
            self.aux2 = self.aux1.clone().t()

    def calc_rep_forces(self, state, velocity_state, goal):
        num_ped = state.shape[0]-1
        if num_ped != self.num_ped:
            self.change_num_of_ped(num_ped)

        pr = self.param.socForcePersonPerson["d"] * \
            torch.ones(num_ped+1, num_ped+1,device=self.device)
        pr[0, :] = self.param.socForceRobotPerson["d"]
        pr[:, 0] = self.param.socForceRobotPerson["d"]
        betta = self.param.socForcePersonPerson["B"] * \
            torch.ones(num_ped+1, num_ped+1,device=self.device)
        betta[0, :] = self.param.socForceRobotPerson["B"]
        betta[:, 0] = self.param.socForceRobotPerson["B"]
        alpha = self.param.socForcePersonPerson["A"] * \
            (1 - torch.eye(self.num_ped+1, self.num_ped+1,device=self.device))
        alpha[0, :] = self.param.socForceRobotPerson["A"]
        alpha[:, 0] = self.param.socForceRobotPerson["A"]

        state_concated = state.clone().matmul(self.aux1)
        state_concated_t = state.reshape(1, -1)

# delta = ((ax1-ax0)*( bx1-bx0)+( ay1-ay0)*( by1-by0))/(| a | * | b|);
# | a | = √[(ax1-ax0 )^2 + ( ay1-ay0)^2]
# k_group =min(delta/self.param.agroup,1)
# f = f*k_group
        for i in range(0, state.shape[0]-1):
            state_concated_t = torch.cat(
                [state_concated_t, state.reshape(1, -1)])
        '''    state_concated_t tensor(
        [[ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
        [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
        [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.],
        [ 1.,  0.,  2.,  1., -1., -1.,  0., -1.]]
        '''
        g_s_diff = (goal-state)
        lens = torch.norm(g_s_diff,dim=1)
        lens_r = lens.unsqueeze(1).repeat(1,lens.shape[0])

        znam = (lens*lens_r)+1e-9
        A1 = g_s_diff.repeat(1,lens.shape[0])
        A2 = g_s_diff.reshape(-1).unsqueeze(0).repeat(lens.shape[0],1)
        A3 = A1*A2
        chis = torch.matmul(A3,self.aux)
        angle_delta = torch.acos(chis/znam)
        angle_delta[angle_delta!=angle_delta]=0
        angle_delta = torch.abs(angle_delta)
        angle_delta = torch.clamp(angle_delta/self.param.group_angle,max=1.)
        angle_delta=(self.aux@angle_delta).T

        delta_pose = (-state_concated_t + state_concated) #+ 1e-6

        dist_squared = delta_pose ** 2
        dist = (dist_squared.matmul(self.aux))
        # aka distance
        temp = torch.eye(dist.shape[0],device=self.device)
        dist = torch.sqrt(dist) + 10000000 * temp  # TODO: deal with 1/0,
        force_amplitude = alpha * torch.exp((pr - dist) / betta)
        force = force_amplitude.matmul(
            self.auxullary)*(delta_pose / (dist).matmul(self.auxullary)) # * anisotropy
        force = (force * ((self.auxullary - 1) * -1))
        # if ((angle_delta>0.2)*(angle_delta<0.7)).any():
        #     print(angle_delta)
        #     pass
        force = force*angle_delta
        force = force.matmul(self.aux2)
        force[force!=force]=0
        return force
