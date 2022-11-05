import torch
import numpy as np
import argparse
import time

from tqdm import tqdm
from pathlib import Path
from pandas import DataFrame

from utils.general import init_model
from dataloader.dataset import create_dataloader


IMG_SIZE = 224 # comes from train.py
CLASSES = {
    0: ('no_object', 1.), # This value should be changed
    1: ('label1', 0.4),
    2: ('label2', 0.6),
    3: ('label3', 0.2),
    4: ('label4', 0.5)
    }


def inference(args: argparse.ArgumentParser) -> None:
    THRESHOLDS = [cls_thr[1] for cls_thr in CLASSES.values()]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'
    
    model = init_model(args.weights, device=device, num_classes=len(CLASSES), img_size=IMG_SIZE)
    if half:
        model.to(device)
        model.half()
    model.eval()
    
    dataloader = create_dataloader(args.input, args.batch_size)
    total = len(dataloader)
    
    img0 = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device)  # init img
    _ = model(img0.half() if half else img0) if device.type != 'cpu' else None  # run once
    img_paths = []
    outputs = []
    for imgs, paths in tqdm(dataloader, total=total):
        img_paths.extend(paths)
        if half:
            imgs = imgs.to(device, non_blocking=True)
            imgs = imgs.half()
        outputs.append(model(imgs))
    
    outputs = torch.cat(outputs, dim=0)
    outputs.sigmoid_()
    outputs[:, 1:] *= (1. - outputs[:, 0:1])
    
    preds = torch.where(outputs >= torch.tensor(THRESHOLDS, device=device), 1, 0)
    preds = torch.where(torch.all(preds == 0, dim=1, keepdim=True), torch.tensor([1] + [0] * (len(THRESHOLDS) - 1), device=device), preds)
    
    outputs = outputs.cpu().numpy()
    preds = preds.cpu().numpy()
    
    results = DataFrame(
        [[name, CLASSES[ind][0], str(np.round(outputs[i, ind], decimals=3))] 
        for (i, name) in enumerate(img_paths)
        for ind in np.where(preds[i, :])[0]],
        columns=['Filename', 'Label', 'Confidence']
    )
    
    output = Path(args.output) / (Path(args.input).name + '_result.csv')
    results.to_csv(str(output), header=True, index=False)
    
    no_detection_names = list(set(img_paths) - set(results['Filename']))
    if no_detection_names:
        raise Exception(f'The number of names in "names" is different from results[Filename] {no_detection_names}')
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model.pt path')
    parser.add_argument('--input', type=str, required=True, help='./inference/images')  # file/folder,
    parser.add_argument('--output', type=str, default='.', help='output folder, it must exist')  # output folder
    parser.add_argument('--batch_size', type=int, default=128, help='output folder')
    args = parser.parse_args()
    print(args)
    
    t1 = time.time()
    with torch.no_grad():
        inference(args)
    t2 = time.time()
    print(f'Time of execution {round(t2-t1, 4)}s')
    return None


if __name__ == '__main__':
    main()