
import torch
from RepulsiveForces import RepulsiveForces


class SFM:
    def __init__(self, param,device='cpu'):
        self.param = param
        self.rep_f = RepulsiveForces(self.param, device=device)
        self.device = device

    def pose_propagation(self, force, state, speed_vect = None):
        DT = self.param.DT
        vx_vy_uncl = state[:, 2:4] + (force[:,:2]*DT)
        dx_dy = state[:, 2:4]*DT + (force[:,:2]*(DT**2))*0.5

        # //apply constrains:
        # torch.sqrt(vx_vy[:,0:1]**2 + vx_vy[:,1:2]**2)
        
        # TODO: switch moving model
        pose_prop_v_unclamped = vx_vy_uncl.norm(dim=1)
        # self.param.robot_speed = speed_vect[0]        
        # speed_vect = None
        if speed_vect == None:
            ps = self.param.pedestrians_speed
            rs = self.param.robot_speed
            pose_prop_v = torch.clamp(pose_prop_v_unclamped, min=-ps, max=ps)
            pose_prop_v[0] = torch.clamp(pose_prop_v_unclamped[0], min=-rs, max=rs) #todo: fix robot speed  double clamp
            vx_vy = torch.clamp(vx_vy_uncl, min=-ps, max=ps)
            vx_vy[0, :] = torch.clamp(vx_vy_uncl[0, :], min=-rs, max=rs)
        else:
            pose_prop_v = torch.zeros_like(pose_prop_v_unclamped)
            vx_vy = torch.zeros_like(vx_vy_uncl)
            for n in range(len(pose_prop_v_unclamped)): #TODO: refactoring
                pose_prop_v[n] = torch.clamp(pose_prop_v_unclamped[n], min=-speed_vect[n], max=speed_vect[n])    
                vx_vy[n] = torch.clamp(vx_vy_uncl[n], min=-speed_vect[n], max=speed_vect[n])

        dr = dx_dy.norm(dim=1)  # torch.sqrt(dx_dy[:,0:1]**2 + dx_dy[:,1:2]**2)
        mask = (pose_prop_v * DT < dr)  # * torch.ones(state.shape[0])

        aa = (1. - (pose_prop_v * DT) / dr).view(-1, 1)
        bb = (dx_dy.t()*mask).t()
        dx_dy = dx_dy.clone() - (bb * aa)
        state[:, 0:2] = state[:, 0:2].clone() + dx_dy
        state[:, 2:4] = vx_vy
        # TODO: add angle propagation
        return state

    def calc_cost_function(self, robot_goal, robot_init_pose, agents_pose, policy=None):
        a = self.param.a
        b = self.param.b
        e = self.param.e
        robot_pose = agents_pose[0, :3].clone()
        robot_speed = agents_pose[0, 3:].clone()
        PG = (robot_pose - robot_init_pose).dot((-robot_init_pose +
                                           robot_goal)/torch.norm(-robot_init_pose+robot_goal))
        # Blame
        # PG = torch.clamp(PG, max = 0.5)
        B = torch.zeros(len(agents_pose), 1, requires_grad=False)
        # if torch.norm(robot_speed) > e:
        agents_speed = agents_pose[:, :2] # why it named agents_speed?
        delta = agents_speed - robot_pose[:2]
        norm = -torch.norm(delta, dim=1)/b
        B = torch.exp(norm)  # +0.5
        # Overall Cost
        # PG = PG/len(agents_pose)
        B = (-a*PG+1*B)
        B = B/len(agents_pose)
        B = torch.clamp(B, min=0.002)
        # if ppp != policy:
        #     print (goal, policy)
        #     ppp = policy
        # print (PG, policy)
        return B

    def calc_forces(self, state, goals, speed_vect = None):
        rep_force = self.rep_f.calc_rep_forces(
            state[:, 0:2], state[:, 2:4],goals)
        attr_force = self.force_goal(state, goals,speed_vect)
        return rep_force, attr_force

    # def calc_group_k(self,state,goal):
    #     lines = torch.tensor([state[:,:2],goal[:,:2])


    def force_goal(self, input_state, goal, speed_vect = None):
        num_ped = len(input_state)
        rt = self.param.socForcePersonPerson["relaxation_time"] * torch.ones(num_ped,device=self.device)
        # rt[0] = self.param.socForceRobotPerson["relaxation_time"]
        rt  = rt.view(-1,1)
        v_desired_x_y = goal[:, 0:2] - input_state[:, 0:2]
        v_desired_ = torch.sqrt(v_desired_x_y.clone()[:, 0:1]**2 + v_desired_x_y.clone()[:, 1:2]**2)
        # self.param.robot_speed = speed_vect[0][0]
        # speed_vect = None
        if speed_vect == None:
            ps = self.param.pedestrians_speed
            rs = self.param.robot_speed
            v_desired_x_y[1:] *= ps / v_desired_[1:]
            v_desired_x_y[0] *= rs / v_desired_[0]
        else:
            v_desired_x_y *=speed_vect.unsqueeze(1)/v_desired_
            # TODO: calc speed
        # print (pedestrians_speed)
        F_attr =  rt*(v_desired_x_y - input_state[:, 2:])
        F_attr[F_attr!=F_attr]=0
        return F_attr