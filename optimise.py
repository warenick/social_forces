import torch
from pathlib import Path
from evaluate import eval_file
from SFM_AI import SFM_AI
from ChangedParam import ChangedParam
from gp_no_map import AttGoalPredictor, FullAttGoalPredictor, LSTM_simple
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
    gp_model = LSTM_simple()
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
            sfile.write("\n\t ade "+f'{ades:.3f}'+"\t fde "+f'{fdes:.3f}'+"\t dist "+f'{dists:.3f}'+"\t error "+f'{error:.3f}')
            sfile.write("\n\n")
            
    return error

if __name__ == '__main__':

    # param_list = {'ped_mass': 80., 'pedestrians_speed': 1.02, 'k': 2.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  4.820
#     param_list = {'ped_mass': 80., 'pedestrians_speed': 0.70, 'k': 2.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  5.821
#     param_list = {'ped_mass': 80., 'pedestrians_speed': 1.20, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  3.669
#     param_list = {'ped_mass': 80., 'pedestrians_speed': 1.40, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  3.016
#     param_list = {'ped_mass': 80., 'pedestrians_speed': 1.50, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error 2.939
    # param_list = {'ped_mass': 80., 'pedestrians_speed': 1.60, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.743
#     param_list = {'ped_mass': 80., 'pedestrians_speed': 1.70, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.666
    # param_list = {'ped_mass': 100., 'pedestrians_speed': 1.70, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.614
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.544
    # param_list = {'ped_mass': 140., 'pedestrians_speed': 1.70, 'k': 3.33, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
#  error  2.654
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 3.9, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.321
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.2, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error 2.172
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.58, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.160
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.66, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.133
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.066
# reference error  2.152
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.78, 'A': 3.69, 'B': 0.77, 'd': 0.63}
# error  2.152
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 4.4, 'B': 0.77, 'd': 0.63}
# error  2.308
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 3.0, 'B': 0.77, 'd': 0.63}
# error  1.890
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 2.4, 'B': 0.77, 'd': 0.63}
# error  1.711
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 2.0, 'B': 0.77, 'd': 0.63}
# error  1.532
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 1.7, 'B': 0.77, 'd': 0.63}
# error  1.461
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 1.4, 'B': 0.77, 'd': 0.63}
# error  1.333
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 1.1, 'B': 0.77, 'd': 0.63}
# error  1.249
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.9, 'B': 0.77, 'd': 0.63}
# error  1.178
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.7, 'B': 0.77, 'd': 0.63}
# error  1.110
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.5, 'B': 0.77, 'd': 0.63}
# error  1.068
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.77, 'd': 0.63}
# error  1.042
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.9, 'd': 0.63}
# error  1.052
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.6, 'd': 0.63}
# error  1.033
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.4, 'd': 0.63}
# error  1.029
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.25, 'd': 0.63}
# error  1.043
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.4, 'd': 0.8}
# error  1.052
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.4, 'd': 0.55}
# error  1.021
#     param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.4, 'd': 0.35}
# error  1.023
   
    param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.4, 'd': 0.45}
# error  1.017
# controll error ade  0.681      fde 0.287       dist  0.000     error  0.968
# evaluate error ade 0.842	 fde1.091	 dist 1.032	 error 1.933


    param = [
            param_list["ped_mass"],
            param_list["pedestrians_speed"],
            param_list["k"],
            param_list["lambda"],
            param_list["A"],
            param_list["B"],
            param_list["d"]
            ]
    # opt_fun(param)
    res = minimize(opt_fun, param, method='nelder-mead', bounds=(1e-6,1e3),
                   options={"maxiter":1000,"adaptive":True,'xatol':1e-8,'disp':True})
    print(res.x)
    

    exit()
