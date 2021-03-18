import torch
from pathlib import Path
from evaluate import eval_file
from SFM_AI import SFM_AI
from ChangedParam import ChangedParam
from gp_no_map import AttGoalPredictor, FullAttGoalPredictor
from scipy.optimize import minimize

def opt_fun(param): 
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    path_ = "pedestrian_forecasting_dataloader/data/test/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    # gp prediction
    # gp_model = AttGoalPredictor()
    gp_model = FullAttGoalPredictor()
    gp_model.eval()
    gp_model = gp_model.to(dev)
    gp_model_path = "gp_model.pth"
    checkpoint = torch.load(gp_model_path)
    gp_model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        with open("optimise_data.txt","a") as sfile:
            
            param_list = {
                "ped_mass": param[0],
                "pedestrians_speed": param[1],
                "k":param[2], 
                "lambda": param[3],
                "A":param[4], 
                "B":param[5], 
                "d":param[6] 
                }  
            param = ChangedParam(param_list, dev)
            sfm_ai = SFM_AI(param,dev)
            ades = []
            fdes = []
            dists = []
            print("\nparam list\n",param_list)
            sfile.write("\nparam list\n"+str(param_list))
            
            for file in files:
                ade, fde, dist = eval_file(path_,[file],sfm_ai,gp_model,dev)
                print("\nfile ",file,"\n\t ade ",f'{ade:.3f}',"\t fde ",f'{fde:.3f}', "\t dist ",f'{dist:.3f}')
                sfile.write("\nfile "+str(file)+"\n\t ade "+f'{ade:.3f}'+"\t fde "+f'{fde:.3f}'+"\t dist "+f'{dist:.3f}')
                ades.append(ade)
                fdes.append(fde)
                dists.append(dist)
            
            ades = sum(ades)/len(ades)
            fdes = sum(fdes)/len(fdes)
            dists = sum(dists)/len(dists)

            error = ades+fdes
            print("\n=================================full===============================================")
            print("\t ade ",f'{ades:.3f}', "\t fde",f'{fdes:.3f}', "\t dist ",f'{dists:.3f}',"\t error ",f'{error:.3f}')
            print("\n\n")
            sfile.write("\n=================================full===============================================")
            sfile.write("\n\t ade "+f'{ades:.3f}'+"\t fde"+f'{fdes:.3f}'+"\t dist "+f'{dists:.3f}'+"\t error "+f'{error:.3f}')
            sfile.write("\n\n")
            
    return error

if __name__ == '__main__':

    param_list = {
        "ped_mass": 60,
        "pedestrians_speed": 1.0,
        "k":2.3,
        "lambda":0.59,
        "A":3.66,
        "B":0.79,
        "d":0.65
        }
    param = [
        param_list["ped_mass"],
        param_list["pedestrians_speed"],
        param_list["k"],
        param_list["lambda"],
        param_list["A"],
        param_list["B"],
        param_list["d"]

    ]
    res = minimize(opt_fun,param,method='nelder-mead',bounds=(1e-6,1e3) ,options={"maxiter":1000,"adaptive":True,'xatol':1e-8,'disp':True})
    
    print(res.x)

    exit()
