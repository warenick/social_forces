import torch


class Param:
    def __init__(self, device='cpu', param_list):
        self.device = device
        self.loop_rate = 30.
        self.lr = 1e-4

        self.num_ped = 5
        
        # DISCTIB COEFFICIENTS
        self.goal_std_coef = 4.5
        self.pose_std_coef = 1.0
        self.velocity_std_coef = 2.0

        self.area_size = 10
        self.robot_init_pose = torch.tensor(([1.5, 2.0]), device = self.device)
        
        self.DT = 0.4
        # social force params
        pl = param_list
        self.ped_mass = pl
        self.pedestrians_speed = pl
        self.robot_speed = pl
        self.socForceRobotPerson = {"k": pl, "lambda": pl, "A": pl, "B": pl, "d": pl}
        self.socForcePersonPerson = socForceRobotPerson
        # self.socForcePersonPerson = {"k":5.5, "lambda":1.5, "A":8., "B":0.4,"d":0.01}
        # self.ped_mass = 60
        # self.pedestrians_speed = 1.0
        # self.robot_speed = 1.20
        # self.socForceRobotPerson = {"k": 2.3, "lambda": 0.59, "A": 3.66, "B": 0.79, "d": 0.65}
        # self.socForcePersonPerson = {"k": 4.9, "lambda": 1., "A": 12., "B": 0.64, "d": 0.26}
        
        # cost param
        self.a = 0.025
        self.b = 1  #
        self.e = 0.001  # min speed fo blame

        # self.generateMatrices()
        self.init_calcs()
        self.robot_goal = self.goal[0, 2:4]

    def update_num_ped(self, num_ped):
        self.num_ped = num_ped
        # self.generateMatrices()        
        self.init_calcs()
        self.robot_goal = self.goal[0, 2:4]

    def init_calcs(self):
        self.goal_mean = self.area_size * torch.rand((self.num_ped, 2),device = self.device)
        self.goal_std = self.goal_std_coef * torch.rand((self.num_ped, 2),device = self.device)

        self.goal_distrib = torch.distributions.normal.Normal(self.goal_mean, self.goal_std)

        self.goal = self.goal_mean
        self.goal = self.goal.view(-1, 2)

        self.input_state_mean = self.area_size * torch.rand((self.num_ped, 4),device = self.device)
        self.input_state_mean[:, 2:4] = self.input_state_mean[:, 2:4] / self.area_size

        self.input_state_std = self.pose_std_coef * torch.rand((self.num_ped, 4),device = self.device)
        self.input_state_std[:, 2:4] = self.velocity_std_coef * torch.rand((self.num_ped, 2),device = self.device)
        self.input_distrib = torch.distributions.normal.Normal(self.input_state_mean, self.input_state_std)

        # self.input_state = self.input_distrib.sample()
        self.input_state = self.input_state_mean
        self.input_state[0, 0:2] = self.robot_init_pose  # .clone().detach()
        self.input_state = self.input_state.view(-1, 4).requires_grad_(True)

if __name__ == "__main__":
    p = Param()
    print("p.input_state_mean", p.input_state_mean)
    print("p.input_state_std ", p.input_state_std)
    print(p.input_state)
    m = torch.distributions.normal.Normal(torch.tensor([0.0, 5.2],device = self.device), torch.tensor([1.0, 0.02],device = self.device))
    t = m.sample()  # normally distributed with loc=0 and scale=1
    print(t)