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
                "k_calc_speed":param[4]
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
   
    # param_list = {'ped_mass': 120., 'pedestrians_speed': 1.70, 'k': 4.5, 'lambda': 0.72, 'A': 0.3, 'B': 0.4, 'd': 0.45}
# error  1.017
# controll error ade  0.681      fde 0.287       dist  0.000     error  0.968
# evaluate error ade 0.842	 fde1.091	 dist 1.032	 error 1.933

# evaluate error
# param_list = {'ped_mass': 126.02394708690218, 'pedestrians_speed': 1.4349823307044705, 'k': 4.303349200317465, 'lambda': 0.7609956442123378, 'A': 0.2879278727514234, 'B': 0.4750446501642508, 'd': 0.42413189029690784}
# file eth_hotel/eth_hotel.txt
#     ade 0.596  fde 0.873  dist 0.799
# file biwi_eth/biwi_eth.txt
#     ade 0.811  fde 1.334  dist 1.189
# file UCY/zara03/zara03.txt
#     ade 0.710  fde 1.154  dist 1.138
# file UCY/students03/students03.txt
#     ade 0.946  fde 1.307  dist 1.264
# file UCY/students01/students01.txt
#     ade 0.811  fde 1.010  dist 0.942
# file UCY/zara01/zara01.txt
#     ade 0.716  fde 1.068  dist 1.030
# file UCY/zara02/zara02.txt
#     ade 0.630  fde 0.940  dist 0.864
# =================================full===============================================
#     ade 0.746  fde 1.098  dist 1.032     error 1.844

    # param_list = {'relaxation_time': 0.5, 'lambda': 0.761, 'A': 0.288, 'B': 0.475, 'd': 0.424,"k_calc_speed":1.3}
# error  1.531
# evaluate
# file eth_hotel/eth_hotel.txt
# 	 ade 0.164	 fde 0.314	 dist 0.357	 adeS ['0.025', '0.049', '0.070', '0.094', '0.120', '0.147', '0.171', '0.202', '0.232', '0.254', '0.284', '0.314']
# file UCY/students01/students01.txt
# 	 ade 0.568	 fde 1.158	 dist 0.995	 adeS ['0.036', '0.101', '0.185', '0.277', '0.378', '0.486', '0.600', '0.717', '0.838', '0.962', '1.075', '1.158']
# file UCY/zara03/zara03.txt
# 	 ade 0.450	 fde 0.785	 dist 1.034	 adeS ['0.031', '0.080', '0.135', '0.207', '0.295', '0.397', '0.509', '0.619', '0.723', '0.825', '0.797', '0.785']
# file UCY/zara02/zara02.txt
# 	 ade 0.573	 fde 1.064	 dist 1.169	 adeS ['0.031', '0.091', '0.172', '0.270', '0.380', '0.499', '0.625', '0.755', '0.890', '1.029', '1.067', '1.064']
# file UCY/zara01/zara01.txt
# 	 ade 0.485	 fde 1.039	 dist 1.012	 adeS ['0.025', '0.071', '0.135', '0.210', '0.296', '0.390', '0.491', '0.604', '0.729', '0.864', '0.962', '1.039']
# file UCY/students03/students03.txt
# 	 ade 0.772	 fde 1.587	 dist 1.538	 adeS ['0.048', '0.132', '0.241', '0.368', '0.507', '0.655', '0.813', '0.978', '1.147', '1.320', '1.465', '1.587']
# file biwi_eth/biwi_eth.txt
# 	 ade 0.653	 fde 1.107	 dist 1.197	 adeS ['0.112', '0.199', '0.284', '0.386', '0.486', '0.596', '0.703', '0.828', '0.947', '1.080', '1.110', '1.107']
# =================================full===============================================
# 	 ade 0.523	 fde 1.008	 dist 1.043
# 	 adeS ['0.000', '0.000', '0.000', '0.000']	 error 1.531
    # param_list = {'relaxation_time': 0.6, 'lambda': 0.761, 'A': 0.288, 'B': 0.475, 'd': 0.424,"k_calc_speed":1.3}
# error  1.516  ade 0.527	 fde 0.989

    # param_list = {'relaxation_time': 0.6, 'A': 0.288, 'B': 0.475, 'd': 0.424, 'k_calc_speed': 1.1550000000000002}
# ade  0.483 	 fde 0.988 	 dist  1.043 	 error  1.472

# param_list = {'relaxation_time': 0.6168000000000001, 'A': 0.28224, 'B': 0.48829999999999996, 'd': 0.40619200000000005, 'k_calc_speed': 1.1307999999999998}
# ade  0.480 	 fde 0.991 	 dist  1.043 	 error  1.471

#  {'relaxation_time': 0.6201600000000002, 'A': 0.28108799999999995, 'B': 0.49095999999999995, 'd': 0.39839040000000014, 'k_calc_speed': 1.1369599999999997}
# ade  0.480 	 fde 0.989 	 dist  1.043 	 error  1.469 

#  {'relaxation_time': 0.6262049546839991, 'A': 0.2685537495982632, 'B': 0.4820336806342723, 'd': 0.40453868423776357, 'k_calc_speed': 1.1374911502870644}
# ade  0.480 	 fde 0.988 	 dist  1.043 	 error  1.468 

# {'relaxation_time': 0.6409534538843576, 'A': 0.26751954271408657, 'B': 0.450400810048443, 'd': 0.4028951469528649, 'k_calc_speed': 1.150263510959664}
# ade  0.481 	 fde 0.986 	 dist  1.043 	 error  1.467 

#  {'relaxation_time': 0.6403114103589792, 'A': 0.2660103220654016, 'B': 0.48145474343690847, 'd': 0.3788541343441028, 'k_calc_speed': 1.1387629731872857}
# ade  0.479 	 fde 0.987 	 dist  1.043 	 error  1.466

# {'relaxation_time': 0.6480151156105425, 'A': 0.25140733633695017, 'B': 0.4586442372420759, 'd': 0.38009178561428525, 'k_calc_speed': 1.1521234794478272}
# ade  0.480 	 fde 0.985 	 dist  1.043 	 error  1.465

#  {'relaxation_time': 0.6748482233152864, 'A': 0.25364815264949114, 'B': 0.45375331267670693, 'd': 0.36302762830307655, 'k_calc_speed': 1.1338628533738007}
# ade  0.478 	 fde 0.987 	 dist  1.043 	 error  1.465 

# {'relaxation_time': 0.6703549400349506, 'A': 0.24294689871705533, 'B': 0.4746347085329762, 'd': 0.32748176861889444, 'k_calc_speed': 1.1433906513000822}
# ade  0.478 	 fde 0.985 	 dist  1.043 	 error  1.463 

# {'relaxation_time': 0.687417792729869, 'A': 0.22946305056420663, 'B': 0.45471877520591575, 'd': 0.33839672715975266, 'k_calc_speed': 1.1444958586329532}
# ade  0.478 	 fde 0.985 	 dist  1.043 	 error  1.462 

# {'relaxation_time': 0.6912436261865784, 'A': 0.219038308350898, 'B': 0.4544626847999328, 'd': 0.3056360502175941, 'k_calc_speed': 1.1514701872949793}
# ade  0.478 	 fde 0.983 	 dist  1.043 	 error  1.461 

    # param_list = {'relaxation_time': 0.7697202735196284, 'A': 0.16202029569194099, 'B': 0.4511158078800178, 'd': 0.19784128285704994, 'k_calc_speed': 1.157552156679437}
# ade  0.478 	 fde 0.981 	 dist  1.043 	 error  1.459 










    param_list =  {'relaxation_time': 0.7994007050033303, 'A': 0.1532433321405573, 'B': 0.45022516823729253, 'd': 0.19791056072000723, 'k_calc_speed': 1.147949527066148}
# ade  0.476 	 fde 0.982 	 dist  1.043 	 error  1.458 




    param = [
            param_list["relaxation_time"],
            param_list["A"],
            param_list["B"],
            param_list["d"],
            param_list["k_calc_speed"]
            ]
    # opt_fun(param)
    res = minimize(opt_fun, param, method='nelder-mead', bounds=(1e-6,1e3),
                   options={"maxiter":1000,"adaptive":True,'xatol':1e-8,'disp':True})
    print(res.x)
    

    exit()
