import sys
import os
import torch
from SFM_AI import SFM_AI
from pathlib import Path
from pedestrian_forecasting_dataloader.dataloader import DatasetFromTxt, collate_wrapper
from pedestrian_forecasting_dataloader.config import cfg
from gp_no_map import AttGoalPredictor, LSTM_simple
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

cfg["raster_params"]["draw_hist"] = False
cfg["raster_params"]["use_map"] = False
cfg["raster_params"]["normalize"] = True
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


def eval_file(path_,file,sfm_ai,gp_model,dev='cpu'):
    print("used files: ",file[0])
    dataset = DatasetFromTxt(path_, file, cfg)
    print("len dataset:", len(dataset))
    # dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=12, collate_fn=collate_wrapper, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_wrapper, pin_memory=True)

    pbar = tqdm(dataloader)
    ades = []
    fdes = []
    dists = []
    for batch_num, data in enumerate(pbar):
        # if batch_num > 0.1*len(dataloader):
        #     break
        if np.sum(data.tgt_avail[:, -1]) == 0:
            print("no one have goal")
            continue

        b_mask = (np.sum(data.tgt_avail,axis=1)==12)*(np.sum(data.history_av,axis=1)==8)
        self_poses = torch.tensor(data.history_positions,device=dev,dtype=torch.float)[b_mask] #
        bs = self_poses.shape[0]
        if bs<1:
            print("no one have full history batch")
            continue
        num_peds = 0 #
        i=0 #
        for sequence in data.history_agents:
            if (b_mask[i]): #
                num_peds = max(num_peds, len(sequence))
            i=i+1 #

        neighb_poses = torch.zeros((bs, num_peds, 8, 6),device=dev)
        neighb_poses_avail = torch.zeros((bs, num_peds, 8),device=dev)
        for i in range(len(data.history_agents)):
            if (b_mask[i]):
                for j in range(num_peds):
                    try:
                        neighb_poses[i, j] = torch.tensor(data.history_agents[i][j],device=dev,dtype=torch.float)
                        neighb_poses_avail[i, j] = torch.tensor(data.history_agents_avail[i][j],device=dev,dtype=torch.float)
                    except:
                        # ????
                        pass

        gt_goals = torch.tensor(data.tgt[:, -1, :],device=dev,dtype=torch.float)[b_mask] #
        traj_tgt = torch.tensor(data.tgt,device=dev,dtype=torch.float)[b_mask] #
        predictions = gp_model(self_poses, neighb_poses)
        mean_poses, _ = sfm_ai.get_sfm_predictions(
            agent_state=self_poses[:, 0, :2],
            neighb_state=neighb_poses[:, :, 0, :2] + 0.05 * torch.rand_like(neighb_poses[:, :, 0, :2] ,device=dev),
            agent_vel=self_poses[:, 0, 2:4], 
            neighb_vel=neighb_poses[:, :, 0, 2:4],
            # agent_goal=gt_goals+ 0.05 * torch.rand_like(gt_goals),
            agent_goal=predictions+ 0.05 * torch.rand_like(gt_goals),
            neighb_goal=lin_model(neighb_poses, neighb_poses_avail), num_threads=0
        )
        if len(mean_poses[mean_poses != mean_poses]) != 0:
            print("sfm nans!")
            continue
        mask_goal = torch.tensor(data.tgt_avail[:, -1],device=dev,dtype=torch.bool)[b_mask] #
        mask_traj = torch.tensor(data.tgt_avail,device=dev,dtype=torch.bool)[b_mask] #

        ades.append(ade_loss(mean_poses[:, :, :2].detach(), traj_tgt, mask_traj).item())
        dists.append(distance(predictions, gt_goals, mask_goal))
        fdes.append(distance(mean_poses[:,-1,:2], gt_goals, mask_goal))
        pbar.set_postfix({' bs ': bs," num peds ":num_peds})
        # print('bs', bs)
    pbar.close()
    try:
        ade = sum(ades)/len(ades)
        fde = (sum(fdes)/len(fdes)).tolist()
        dist = (sum(dists)/len(dists)).tolist()
    except:
        print("dividing by zero")
        ade = 10
        fde = 10
        dist = 10

    return ade, fde, dist



if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    # init SFM
    sfm_ai = SFM_AI(device=dev)
    # gp prediction
    gp_model = LSTM_simple()
    gp_model.eval()
    gp_model = gp_model.to(dev)
    gp_model_path = "gp_model.pth"
    checkpoint = torch.load(gp_model_path)
    gp_model.load_state_dict(checkpoint['model_state_dict'])

    path_ = "pedestrian_forecasting_dataloader/data/test/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    with torch.no_grad():
        for file in files:
            ade, fde, dist = eval_file(path_,[file],sfm_ai,gp_model,dev)
            print("\nfile ",file,"\n\t ade ",ade,"\t fde ",fde,"\t dist ",dist)
    exit()

    # "biwi_eth/biwi_eth.txt"
# {'mean_ade': 1.6717669055575417, 'mean_fde': 2.8545318898700534, 'mean_dist': 2.2019563913345337}

# "crowds/crowds_zara02.txt"
# {'mean_ade': 2.058896233295572, 'mean_fde': 2.310119941316802, 'mean_dist': 1.9428932748991867}

# "crowds/students001.txt"
# {'mean_ade': 3.285138066135236, 'mean_fde': 4.164177453339989, 'mean_dist': 1.634656242470243}

# "crowds/students003.txt"
# {'mean_ade': 2.8972231082196505, 'mean_fde': 3.465533395983138, 'mean_dist': 1.6947245597839355}