import sys
import os
import torch
from SFM_AI import SFM_AI
from pathlib import Path
from pedestrian_forecasting_dataloader.dataloader import DatasetFromTxt, collate_wrapper
from pedestrian_forecasting_dataloader.config import cfg
from gp_no_map import LSTM_simple
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from ChangedParam import ChangedParam
from gp_no_map import LSTM_simple



cfg["raster_params"]["draw_hist"] = False
cfg["raster_params"]["use_map"] = False
cfg["raster_params"]["normalize"] = True
def lin_model(neigb_states, neigb_avail):
    goals = neigb_states[:, :, 0, 2:4] * 12 + neigb_states[:, :, 0, :2]
    goals[(neigb_avail[:, :, 1] == 0).bool()] = neigb_states[:, :, 0, :2][(neigb_avail[:, :, 1] == 0).bool()]
    return goals


def eval_file(path_,file,sfm_ai,gp_model,dev='cpu'):
    dataset = DatasetFromTxt(path_, file, cfg)
    print("used files: ",file[0])
    print("len dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_wrapper, pin_memory=True)
    pbar = tqdm(dataloader)
    forces = []
    for batch_num, data in enumerate(pbar):
        # if batch_num > 0.1*len(dataloader):
        #     break
        # if np.sum(data.tgt_avail[:, -1]) == 0:
        #     print("no one have full future in batch")
        #     continue

        # b_mask = (np.sum(data.tgt_avail,axis=1)==12)*(np.sum(data.history_av,axis=1)==8)
        self_poses = torch.tensor(data.history_positions,device=dev,dtype=torch.float)#[b_mask] #
        bs = self_poses.shape[0]
        if bs<1:
            print("no one have full history in batch")
            continue
        num_peds = 0 #
       # i=0 #
        for sequence in data.history_agents:
           # if (b_mask[i]): #
            num_peds = max(num_peds, len(sequence))
            #i=i+1 #
        neighb_poses = -100*torch.ones((bs, num_peds, 8, 6),device=dev)
        neighb_poses_avail = torch.zeros((bs, num_peds, 8),device=dev)
        for i in range(len(data.history_agents)):
            #if (b_mask[i]):
            for j in range(num_peds):
                try:
                    neighb_poses[i, j] = torch.tensor(data.history_agents[i][j],device=dev,dtype=torch.float)
                    neighb_poses_avail[i, j] = torch.tensor(data.history_agents_avail[i][j],device=dev,dtype=torch.float)
                except:
                    # ????
                    pass
        predictions = gp_model(self_poses, neighb_poses)
        _, F = sfm_ai.get_sfm_predictions(
            agent_state=self_poses[:, 0, :2],
            neighb_state=neighb_poses[:, :, 0, :2],# + 0.0001 * torch.rand_like(neighb_poses[:, :, 0, :2] ,device=dev),
            agent_vel=self_poses[:, 0, 2:4], 
            neighb_vel=neighb_poses[:, :, 0, 2:4],
            agent_goal=predictions,#+ 0.0001 * torch.rand_like(gt_goals),
            neighb_goal=lin_model(neighb_poses, neighb_poses_avail), 
            num_threads=0, 
            calc_speeds=True,
            future_horizon=1
        )
        idwithforce = torch.cat([torch.tensor(data.agent_ts_id[:,0]),F[:,0]],dim=1)
        forces.append(idwithforce)
        # pbar.set_postfix({' bs ': bs," num peds ":num_peds})
    pbar.close()
    # save forces
    pbar = tqdm(forces)
    with open(path_+file[0][:-4]+".force","w") as sfile:
        for line_num, data in enumerate(pbar):
            for scene in data:
                string = ',\t'.join(['%.5f']*len(scene.tolist()))% tuple(scene.tolist())
                sfile.write(string+'\n')

    return 



if __name__ == '__main__':


    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    # SFM
    param_list =  {'relaxation_time': 0.8153887191033968, 'A': -3.15630819878336843, 'B': 0.4592296716020384, 'd': 0.2018687719344074, 'k_calc_speed': 1.0905520507128406,'group_angle':0.524}
    print("\nparam list\n",param_list)            
    param = ChangedParam(param_list, dev)
    sfm_ai = SFM_AI(param,dev)

    # gp prediction
    gp_model = LSTM_simple()
    gp_model.eval()
    gp_model = gp_model.to(dev)
    gp_model_path = "gp_model.pth"
    checkpoint = torch.load(gp_model_path)
    gp_model.load_state_dict(checkpoint['model_state_dict'])

    # data
    path_ = "pedestrian_forecasting_dataloader/data/test/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    files = [
        'SDD/bookstore_6.txt',
        'SDD/nexus_4.txt',
        'SDD/nexus_5.txt',
        'SDD/hyang_10.txt',
        'SDD/little_2.txt',
        'SDD/hyang_2.txt',
        'SDD/bookstore_1.txt',
        'SDD/gates_4.txt',
        'SDD/hyang_5.txt',
        'SDD/gates_5.txt',
        'SDD/hyang_7.txt',
        'SDD/coupa_0.txt',
        'SDD/gates_0.txt',
        'SDD/hyang_0.txt',
        'SDD/nexus_9.txt',
        'SDD/nexus_10.txt',
        'SDD/coupa_2.txt'
    ]
    torch.set_printoptions(precision=5,sci_mode=False)
    with torch.no_grad():
        for file in files:
                eval_file(path_,[file],sfm_ai,gp_model,dev)                

    exit()