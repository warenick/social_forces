import torch
from pathlib import Path
from evaluate import eval_file

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    path_ = "pedestrian_forecasting_dataloader/data/test/"
    # get all availible data
    pathes = list(Path(path_).rglob("*.[tT][xX][tT]"))
    files = [str(x).replace(path_,"") for x in pathes]
    with torch.no_grad():
        for file in files:
            ade, fde = eval_file(path_,[file],dev)
            print("\nfile",file,"\n\t ade ",ade,"\n\t fde ",fde)
    exit()
