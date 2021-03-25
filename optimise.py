import torch
from pathlib import Path
from evaluate import eval_file
from SFM_AI import SFM_AI
from ChangedParam import ChangedParam
from gp_no_map import AttGoalPredictor, FullAttGoalPredictor, LSTM_simple
from scipy.optimize import minimize
torch.set_printoptions(precision=3,sci_mode=False)

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
                "relaxation_time":param[0], 
                # "lambda": param[1],
                "A":param[1], 
                "B":param[2], 
                "d":param[3],
                "k_calc_speed":param[4],
                "group_angle":param[5]
                }  
                
            param = ChangedParam(param_list, dev)
            sfm_ai = SFM_AI(param,dev)
            ades = []
            fdes = []
            dists = []
            adeS = []
            print("\nparam list\n",param_list)
            sfile.write("\nparam list\n"+str(param_list))
            
            for file in files:
                ade, fde, dist, ade_s = eval_file(path_,[file],sfm_ai,gp_model,dev)
                print("\nfile ",file,"\n\t ade ",f'{ade:.3f}',"\t fde ",f'{fde:.3f}', "\t dist ",f'{dist:.3f}', "\t adeS ",str([f'{x:.3f}' for x in ade_s]))
                sfile.write("\nfile "+str(file)+"\n\t ade "+f'{ade:.3f}'+"\t fde "+f'{fde:.3f}'+"\t dist "+f'{dist:.3f}'+"\t adeS "+str([f'{x:.3f}' for x in ade_s]))
                ades.append(ade)
                fdes.append(fde)
                dists.append(dist)
                adeS.append(ade_s)
            ades = sum(ades)/len(ades)
            fdes = sum(fdes)/len(fdes)
            dists = sum(dists)/len(dists)
            adeS = [0,0,0,0]
            # adeS = torch.stack(adeS).permute(1,0).mean(axis=1).tolist()

            error = ades+fdes
            print("\n=================================full===============================================")
            print("\t ade ",f'{ades:.3f}', "\t fde",f'{fdes:.3f}', "\t dist ",f'{dists:.3f}',"\t error ",f'{error:.3f}', "\n\t adeS ",str([f'{x:.3f}' for x in adeS]))
            print("\n\n")
            sfile.write("\n=================================full===============================================")
            sfile.write("\n\t ade "+f'{ades:.3f}'+"\t fde "+f'{fdes:.3f}'+"\t dist "+f'{dists:.3f}'+"\t error "+f'{error:.3f}'+"\n\t adeS "+str([f'{x:.3f}' for x in adeS]))
            sfile.write("\n\n")
            
    return error

if __name__ == '__main__':

# param list
# {'relaxation_time': 0.8153887191033968, 'A': 0.15630819878336843, 'B': 0.4592296716020384, 'd': 0.2018687719344074, 'k_calc_speed': 1.0905520507128406}
# file eth_hotel/eth_hotel.txt
#     ade 0.386  fde 0.747  dist 0.799     adeS ['0.063', '0.115', '0.163', '0.223', '0.284', '0.345', '0.407', '0.470', '0.538', '0.605', '0.673', '0.747']
# file biwi_eth/biwi_eth.txt
#     ade 0.580  fde 1.127  dist 1.189     adeS ['0.108', '0.187', '0.257', '0.341', '0.424', '0.510', '0.603', '0.696', '0.795', '0.902', '1.011', '1.127']
# file UCY/zara03/zara03.txt
#     ade 0.480  fde 1.055  dist 1.138     adeS ['0.034', '0.080', '0.135', '0.207', '0.292', '0.389', '0.494', '0.603', '0.713', '0.826', '0.936', '1.055']
# file UCY/students03/students03.txt
#     ade 0.597  fde 1.277  dist 1.264     adeS ['0.046', '0.111', '0.192', '0.286', '0.388', '0.498', '0.615', '0.739', '0.867', '1.001', '1.138', '1.277']
# file UCY/students01/students01.txt
#     ade 0.441  fde 0.956  dist 0.942     adeS ['0.032', '0.079', '0.138', '0.206', '0.282', '0.364', '0.453', '0.546', '0.643', '0.745', '0.849', '0.956']
# file UCY/zara01/zara01.txt
#     ade 0.443  fde 0.961  dist 1.030     adeS ['0.033', '0.082', '0.144', '0.215', '0.291', '0.372', '0.456', '0.544', '0.638', '0.736', '0.841', '0.961']
# file UCY/zara02/zara02.txt
#     ade 0.380  fde 0.813  dist 0.864     adeS ['0.028', '0.070', '0.122', '0.182', '0.248', '0.318', '0.393', '0.471', '0.552', '0.637', '0.724', '0.813']
# =================================full===============================================
#     ade 0.472  fde 0.991  dist 1.032     error 1.463
# ade  0.472      fde 0.990       dist  1.032     error  1.462 

    param_list =  {'relaxation_time': 0.8153887191033968, 'A': -3.15630819878336843, 'B': 0.4592296716020384, 'd': 0.2018687719344074, 'k_calc_speed': 1.0905520507128406,'group_angle':0.524}
# cutted dataset ade  0.475      fde 0.991       dist  1.043     error  1.466




    param = [
            param_list["relaxation_time"],
            param_list["A"],
            param_list["B"],
            param_list["d"],
            param_list["k_calc_speed"],
            param_list['group_angle']
            ]
    opt_fun(param)
    # res = minimize(opt_fun, param, method='nelder-mead', bounds=(1e-6,1e3),
    #                options={"maxiter":1000,"adaptive":True,'xatol':1e-8,'disp':True})
    # print(res.x)
    

    exit()
