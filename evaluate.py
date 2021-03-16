import sys
import os
import torch
from SFM_AI import SFM_AI
from pathlib import Path
from pedestrian_forecasting_dataloader.dataloader import DatasetFromTxt, collate_wrapper
from pedestrian_forecasting_dataloader.config import cfg
from gp_no_map import AttGoalPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def lin_model(neigb_states, neigb_avail):
    goals = neigb_states[:, :, 0, 2:4] * 12 + neigb_states[:, :, 0, :2]
    goals[(neigb_avail[:, :, 1] == 0).bool()] = neigb_states[:, :, 0, :2][(neigb_avail[:, :, 1] == 0).bool()]
    return goals

def ade_loss(pred_traj, gt, mask):
    assert pred_traj.ndim == 3
    assert gt.ndim == 3
    assert mask.ndim == 2
    error = pred_traj - gt
    norm = torch.norm(error, dim=2)[mask]
    return torch.mean(norm)


def distance(prediction, gt, tgt_avail=None):
    if prediction.ndim == 3:
        return oracle_distance(prediction, gt, tgt_avail)
    if tgt_avail is None:
        return torch.mean(torch.sqrt(torch.sum((prediction - gt) ** 2, dim=1)))
    tgt_avail = tgt_avail.bool().to(prediction.device)
    error = prediction - gt
    norm = torch.norm(error, dim=1)
    error_masked = norm[tgt_avail]
    if torch.sum(tgt_avail) != 0:
        return torch.mean(error_masked)
    else:
        print("no gt available?")
        return 0


def eval_file(path_,file,sfm_ai,dev='cpu'):
    print("used files:")
    print(file)
    dataset = DatasetFromTxt(path_, file, cfg)
    print("len dataset:", len(dataset))
    # dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=12, collate_fn=collate_wrapper, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_wrapper, pin_memory=True)


    # gp prediction
    gp_model = AttGoalPredictor()
    gp_model.eval()
    gp_model = gp_model.to(dev)
    gp_model_path = "gp_model.pth"
    checkpoint = torch.load(gp_model_path)
    gp_model.load_state_dict(checkpoint['model_state_dict'])
    # print("gp loaded")
    epoch = 0
    checkpoint = None

    pbar = tqdm(dataloader)
    ades = []
    fdes = []
    for batch_num, data in enumerate(pbar):

        if np.sum(data.tgt_avail[:, -1]) == 0:
            print("no one have goal")
            continue

        self_poses = torch.tensor(data.history_positions,device=dev).float()

        num_peds = 0
        for sequence in data.history_agents:
            num_peds = max(num_peds, len(sequence))

        neighb_poses = torch.zeros((len(data.history_agents), num_peds, 8, 6),device=dev)
        neighb_poses_avail = torch.zeros((len(data.history_agents), num_peds, 8),device=dev)
        for i in range(len(data.history_agents)):
            for j in range(num_peds):
                try:
                    neighb_poses[i, j] = torch.tensor(data.history_agents[i][j],device=dev,dtype=torch.float)
                    neighb_poses_avail[i, j] = torch.tensor(data.history_agents_avail[i][j],device=dev,dtype=torch.float)
                except:
                    # ????
                    pass
        gt_goals = torch.tensor(data.tgt[:, -1, :],device=dev,dtype=torch.float)
        traj_tgt = torch.tensor(data.tgt,device=dev,dtype=torch.float)
        predictions = gp_model(self_poses, neighb_poses)[:, 0, :]
        mean_poses, _ = sfm_ai.get_sfm_predictions(
            agent_state=self_poses[:, 0, :2] + 0.00 * torch.rand_like(self_poses[:, 0, :2],device=dev),
            neighb_state=neighb_poses[:, :, 0, :2] + 0.01 * torch.rand_like(neighb_poses[:, :, 0, :2],device=dev),
            agent_vel=self_poses[:, 0, 2:4], neighb_vel=neighb_poses[:, :, 0, 2:4],
            agent_goal=predictions,
            neighb_goal=lin_model(neighb_poses, neighb_poses_avail), num_threads=6
        )
        if len(mean_poses[mean_poses != mean_poses]) != 0:
            print("sfm nans!")
            continue
        mask_goal = torch.tensor(data.tgt_avail[:, -1],device=dev,dtype=torch.bool)
        mask_traj = torch.tensor(data.tgt_avail,device=dev,dtype=torch.bool)

        ade_metric = ade_loss(mean_poses[:, :, :2].detach(), traj_tgt, mask_traj)
        ades.append(ade_metric.item())
        dist_err = distance(predictions, gt_goals, mask_goal)
        fdes.append(distance(mean_poses[:,-1,:2], gt_goals, mask_goal))
        # pbar.set_postfix({'mean_ade': sum(ades)/len(ades)})
    # print('mean_ade', sum(ades)/len(ades))
    pbar.close()
    ade = sum(ades)/len(ades)
    fde = (sum(fdes)/len(fdes)).data
    return ade, fde



if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    # init SFM
    sfm_ai = SFM_AI(device=dev)

    path_ = "pedestrian_forecasting_dataloader/data/test/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    with torch.no_grad():
        for file in files:
            ade, fde = eval_file(path_,[file],sfm_ai,dev)
            print("\nfile",file,"\n\t ade ",ade,"\n\t fde ",fde)
    exit()

    # "biwi_eth/biwi_eth.txt"
# {'mean_ade': 1.6717669055575417, 'mean_fde': 2.8545318898700534, 'mean_dist': 2.2019563913345337}

# "crowds/crowds_zara02.txt"
# {'mean_ade': 2.058896233295572, 'mean_fde': 2.310119941316802, 'mean_dist': 1.9428932748991867}

# "crowds/students001.txt"
# {'mean_ade': 3.285138066135236, 'mean_fde': 4.164177453339989, 'mean_dist': 1.634656242470243}

# "crowds/students003.txt"
# {'mean_ade': 2.8972231082196505, 'mean_fde': 3.465533395983138, 'mean_dist': 1.6947245597839355}