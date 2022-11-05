import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os
import math
import yaml
import shutil

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from datetime import date
from typing import Dict, Iterator, Tuple, Union, Any, Iterable

from model.cct import CCT


def load_yaml(path: Path):
    with open(path, 'r') as file:
        hyp = yaml.load(file, Loader=yaml.FullLoader)
    return hyp


def write_train_configuration(files: Iterable[Path], dir_save: Path) -> None:
    for file in files:
        shutil.copy(file, dir_save)
    return None


def write_validation_metrics(txt: Path, epoch: int, loss: float, accuracy_mean: float, f1_mean: float, precisions: torch.Tensor, recalls: torch.Tensor):
    if not txt.is_file():
        P_R_head = [f'{metric}{i}' for i in range(recalls.shape[0]) for metric in ['P', 'R']]
        head = ['Epoch', 'Loss ', 'Acc_mean', 'F1_mean'] + P_R_head
        row_format = '{:>8}' * 4 + '{:>8}' * (recalls.shape[0] * 2) + '\n'
        with open(txt, 'w') as file:
            file.write(row_format.format(*head))
            
    row_format = '{:>8}' + '{:>8.4f}' +  '{:>8.3f}' * (2 + recalls.shape[0] * 2) + '\n'
    P_R_head = []
    for i in range(recalls.shape[0]):
        P_R_head.append(precisions[i].item())
        P_R_head.append(recalls[i].item())
    with open(txt, 'a') as file:
        file.write(row_format.format(epoch, loss, accuracy_mean, f1_mean, *P_R_head))

    return None


def create_runs_structure() -> Tuple[Path, Path]:
    today = date.today().strftime("%d-%m-%Y")
    dir_name  = '_'.join(['CCT', today]) 
    
    runs_folder = Path('../runs')
    runs_folder.mkdir(exist_ok=True)
    n = max(
        [int(folder.name.split('_')[-1])
        for folder in runs_folder.iterdir()
        if folder.is_dir() and dir_name in folder.name],
        default=-1)
    n += 1
    exp = dir_name + f'_{n}' # first experiment

    folder_cur = runs_folder / exp

    weights_folder = folder_cur / 'weights'
    tensorboard_dir = folder_cur / 'logs'

    weights_folder.mkdir(parents=True)
    tensorboard_dir.mkdir(parents=True)
    return weights_folder, tensorboard_dir


def print_model_summary(model: nn.Module) -> None:
    param_total = 0
    param_trainable = 0
    for name, param in model.named_parameters():
        num_param = param.numel()
        param_total += num_param
        if param.requires_grad:
            param_trainable += num_param
            print(name, num_param)
        else:
            print(name, 0)
    print(f'Trainable Parameters: {param_trainable:,}; Total: {param_total:,}')
    return None


def freeze_all_layers(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return None


def create_new_head(model: CCT, num_classes: int) -> CCT:
    in_features = model.classifier.fc.in_features
    head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, in_features),
        nn.ReLU(),
        nn.Linear(in_features, num_classes)
    )
    model.classifier.fc = head
    return model


def unfreeze_blocks(model: CCT, num_blocks: int) -> None:
    for block in model.classifier.blocks[-num_blocks: ]:
        for module in block.modules():
            if isinstance(module, nn.LayerNorm):
                continue
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad = True
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad = True
    return None


def set_training_mode_for_blocks(model: CCT, num_blocks: int) -> None:
    for block in model.classifier.blocks[-num_blocks: ]:
        for module in block.modules():
            if isinstance(module, nn.LayerNorm):
                module.eval()
            else:
                module.train()
    return None


def get_oprimizer(opt_name: str, params: Iterator[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt_name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    return optimizer


def define_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int) -> lr_scheduler.LambdaLR:
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / num_epochs)) / 2) ** 1.0) * 0.8 + 0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler


def compute_loss(outputs: torch.Tensor, targets: torch.Tensor, hyp: Dict[str, Any]) -> torch.Tensor:
    viol = targets[:, 0] == 0 # indexes where there is at least one label
    zero_class_weights = None
    viol_class_weights = None
    if hyp['class_weights'] is not None:
        class_weights = torch.tensor(list(hyp['class_weights']))
        zero_class_weights = class_weights[0]
        viol_class_weights = class_weights[1:]

    BCEloss_zero_class = nn.BCEWithLogitsLoss(pos_weight=zero_class_weights).to(targets.device)
    BCEloss_viol_class = nn.BCEWithLogitsLoss(pos_weight=viol_class_weights).to(targets.device)
    
    zero_class_loss = BCEloss_zero_class(outputs[:, 0:1], targets[:, 0:1]) # is responsible for classification "no_object"
    viol_loss = BCEloss_viol_class(outputs[viol][:, 1:], targets[viol][:, 1:]) # is responsible for classification of actual labels

    return zero_class_loss + hyp['loss_class_scaler'] * viol_loss


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs.sigmoid_()
    outputs[:, 1:] *= (1. - outputs[:, 0:1]) # probability of zero class being zero (that is probability of violation) * prob of other classes
    outputs = torch.where(outputs > 0.5, 1, 0)

    tp = torch.count_nonzero(outputs * targets, dim=0)
    fp = torch.count_nonzero(outputs * (1  - targets), dim=0)
    fn = torch.count_nonzero((1 - outputs) * targets, dim=0)
    tn = torch.count_nonzero((1 - outputs) * (1 - targets), dim=0)
    
    return tp, fp, fn, tn


def calculate_training_metrics(running_metrics: Dict[str, torch.Tensor], outputs: torch.Tensor, labels: torch.Tensor) \
        -> Tuple[Dict[str, torch.Tensor], Dict[str, Union[float, torch.Tensor]]] :
    
    batch_metrics = calculate_metrics(outputs, labels)
    for i, metric in enumerate(['tp', 'fp', 'fn', 'tn']): # tp, fp, fn, tn
        running_metrics[metric] += batch_metrics[i]
    
    precisions = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fp'] + 1e-16)
    recalls = running_metrics['tp'] / (running_metrics['tp'] + running_metrics['fn'] + 1e-16)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-16)
    f1_mean = torch.mean(f1s[1:]) # exclude 0-class (no_object (clear screenshot) is excluded)
    
    accuracy = ((running_metrics['tp'] + running_metrics['tn']) / 
                (running_metrics['tp'] + running_metrics['fn'] + running_metrics['fp'] + running_metrics['tn']))
    accuracy_mean = torch.mean(accuracy[1:])
    train_metr = {
        "accuracy_mean": accuracy_mean.item(),
        "f1_mean": f1_mean.item(),
        "precisions": precisions.cpu(),
        "recalls": recalls.cpu(),
    }
    
    return running_metrics, train_metr


def create_print_string(metrics: Dict[str, Union[float, torch.Tensor]]) -> str:
    return '{:>10.4f} {:10.4f}'.format(metrics['accuracy_mean'], metrics['f1_mean'])


def validate(model : CCT, valloader: DataLoader, hyp: Dict[str, Any], num_unfreezed_blocks: int = 0) -> Dict[str, Union[float, torch.Tensor]]:
    device = next(model.parameters()).device
    
    model.half()
    model.eval()
    loss = 0
    tp, fp, fn, tn, num_samples = [torch.zeros(model.num_classes, device=device) for _ in range(5)]
    num_batches = len(valloader)
    for imgs, labels in tqdm(valloader, desc=None, total=num_batches):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs = imgs.half()

        # inference
        outputs = model(imgs)
        loss += compute_loss(outputs, labels, hyp).item()
        # Metrics
        metrics = calculate_metrics(outputs, labels)
        tp += metrics[0]
        fp += metrics[1]
        fn += metrics[2]
        tn += metrics[3]
        num_samples += labels.sum(axis=0)
    
    precisions = tp / (tp + fp + 1e-16)
    recalls = tp / (tp + fn + 1e-16)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-16)
    f1_mean = torch.mean(f1s[1:]) # exclude 0-class (clear screenshot is excluded)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    accuracy_mean = torch.mean(accuracy[1:])
    
    
    model.float()
    model.classifier.fc.train()
    if num_unfreezed_blocks != 0:
        set_training_mode_for_blocks(model, num_unfreezed_blocks)
    
    val_metrs = {
        "loss": loss / num_batches,
        "accuracy_mean": accuracy_mean.item(),
        "f1_mean": f1_mean.item(),
        "precisions": precisions.cpu(),
        "recalls": recalls.cpu(),
        "num_samples": num_samples.cpu()
    }
    
    return val_metrs


def print_metrics(loss: float, accuracy_mean: float, f1_mean:float, precisions: torch.Tensor, recalls: torch.Tensor, num_samples: torch.Tensor) -> None:
    print(f'\nLoss: {loss:0.5f}; Accuracy mean: {accuracy_mean:0.4f}; F1 mean: {f1_mean:0.3f}')
    head = ['Class', 'Total', 'P', 'R']
    row_format = '{:<5} {:<5} {:<5} {:<6} {:<6}'
    print(row_format.format("", *head))
    row_format = '{:<5} {:<5} {:<5} {:>0.3f} {:>0.3f}'

    for values in zip(range(num_samples.shape[0]), num_samples.tolist(), precisions.tolist(), recalls.tolist()):
        print(row_format.format("", *values))
    
    return None


def write_metrics2tensorboard(
        tb_writer: SummaryWriter,
        train_metrics: Dict[str, Union[float, torch.Tensor]],
        val_metrics: Dict[str, Union[float, torch.Tensor]],
        mean_loss: float,
        epoch: int,
        num_blocks: int,
        num_base_epochs: int) -> None:
    if tb_writer is None:
        return None
    
    zero_epoch = 0 if num_blocks == 0 else num_base_epochs
    write_epoch = zero_epoch + epoch
    tb_writer.add_scalars(
        'Train-Val Loss',
        {
            "train": mean_loss,
            "val": val_metrics['loss']
        },
        write_epoch)
    tb_writer.add_scalars(
        'Mean Metrics',
        {
            "accuracy_mean_train": train_metrics['accuracy_mean'],
            "accuracy_mean_val": val_metrics['accuracy_mean'],
            "f1_mean_train": train_metrics['f1_mean'],
            "f1_mean_val": val_metrics['f1_mean']
        },
        write_epoch)
    for i in range(val_metrics['precisions'].shape[0]):
        tb_writer.add_scalars(
            f'P{i}-R{i}',
            {
                f'P{i}_train': train_metrics['precisions'][i].item(),
                f'P{i}_val': val_metrics['precisions'][i].item(),
                f'R{i}_train': train_metrics['recalls'][i].item(),
                f'R{i}_val': val_metrics['recalls'][i].item(),
            },
            write_epoch)
    return None

def init_model(weights: str, device: torch.device, num_classes: int, img_size: int) -> CCT:
    weights = str(Path(weights))

    # These default values come from train.py. 
    kernel_size = 7
    stride = max(1, (kernel_size // 2) - 1)
    padding = max(1, (kernel_size // 2))
    model = CCT(
        kernel_size=kernel_size,
        n_conv_layers=2,
        num_layers=14,
        num_heads=6,
        mlp_ratio=3,
        embed_dim=384,
        stride=stride,
        padding=padding,
        img_size=img_size

    )
    in_features = model.classifier.fc.in_features
    head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, in_features),
        nn.ReLU(),
        nn.Linear(in_features, num_classes)
    )
    model.classifier.fc = head
    missing_keys = model.load_state_dict(torch.load(weights, map_location=device), strict=True)
    print(missing_keys)
    return model
