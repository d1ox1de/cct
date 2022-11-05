import torch
import torch.optim as optim
import argparse

from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict

from model.cct import CCT, cct
from dataloader.dataset import create_dataset_dataloader
from utils.general import (
    load_yaml, freeze_all_layers, unfreeze_blocks, create_new_head,
    get_oprimizer, define_scheduler, compute_loss, validate, print_metrics,
    calculate_training_metrics, create_print_string, print_model_summary,
    create_runs_structure, write_train_configuration, write_validation_metrics,
    write_metrics2tensorboard)

torch.manual_seed(0)


def train(hyp: Dict[str, Any], cnfg: Dict[str, Any], device: torch.device, weights_dir: Path, tb_writer: SummaryWriter) -> None:
    print('Creating train dataloader')
    dataloader, dataset = create_dataset_dataloader(cnfg['csv_train'], hyp, cnfg['batch_size'], is_augment=True, shuffle=True, is_balanced_sampler=hyp['is_balanced_sampler'])
    print('Creating val dataloader')
    valloader, _ = create_dataset_dataloader(cnfg['csv_val'], hyp, cnfg['batch_size'])
    num_classes = dataset.get_num_classes()
    del dataset
    #init model
    model = cct(cnfg['weights']) # load pretrained model
    model.num_classes = num_classes
    freeze_all_layers(model)
    
    #Head training
    model = create_new_head(model, num_classes)
    model.to(device)
    print_model_summary(model)
    print('Starting training the head')
    optimizer = get_oprimizer(cnfg['optimizer'], model.parameters(), cnfg['lr'])
    fit_function(model, dataloader, valloader, optimizer, cnfg['epochs'], weights_dir, tb_writer, hyp=hyp, cnfg=cnfg)
    
    print('Starting Finetunning')
    # Transer learning
    unfreeze_blocks(model, cnfg['num_blocks'])
    print_model_summary(model)
    optimizer_ft = get_oprimizer(cnfg['optimizer'], model.parameters(), cnfg['lr_ft'])
    fit_function(model, dataloader, valloader, optimizer_ft, cnfg['epochs_ft'], weights_dir, tb_writer, hyp=hyp, cnfg=cnfg, num_blocks=cnfg['num_blocks'])
    return None

def fit_function(
        model: CCT,
        dataloader: DataLoader, 
        valloader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        epochs: int, 
        weights_save_dir: Path, 
        tb_writer: SummaryWriter, 
        hyp: Dict[str, Any], 
        cnfg: Dict[str, Any], 
        num_blocks: int = 0, 
        start_epoch: int = 0) -> None:

    device = next(model.parameters()).device
    is_cuda_available = device.type != 'cpu'
    scheduler = define_scheduler(optimizer, epochs)
    
    num_batches = len(dataloader) 
    scaler = amp.GradScaler(enabled=True)
    
    wstemp = weights_save_dir / weights_save_dir.parent.name # weights stemp
    best_path =  str(wstemp) + '_best.pt'
    best_val_loss = float('inf')
    for epoch in range(start_epoch, epochs):
        last_path = str(wstemp) + f'_last_{epoch}.pt'
        prev_last =  str(wstemp) + f'_last_{epoch - 1}.pt'
        
        print(('\n' + '%10s' * 5) % ('Epoch', 'gpu_mem', 'loss', 'acc_mean', 'f1_mean'))
        pbar = tqdm(enumerate(dataloader), total=num_batches)
        
        mean_loss = 0
        running_metrics = {k: torch.zeros(model.num_classes, device=device, requires_grad=False) for k in ['tp', 'fp', 'fn', 'tn']} 
        optimizer.zero_grad()
        for i, (imgs, labels) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp.autocast(enabled=is_cuda_available):
                outputs = model(imgs)
                # be careful about distributed training. Check loss scaling by the batch size
                loss = compute_loss(outputs, labels, hyp)
            
            with torch.no_grad():
                running_metrics, train_metrics = calculate_training_metrics(running_metrics, outputs, labels)
                string_metrics = create_print_string(train_metrics)
            #Backward
            scaler.scale(loss).backward()
            
            #Optimize
            scaler.step(optimizer)
            scaler.update() # Updates the scale for next iteration.
            optimizer.zero_grad()
            
            #printing stats
            mean_loss = (mean_loss * i + loss.item()) / (i + 1)
            mem = '{:.3f}G'.format(torch.cuda.memory_reserved() / 1000**3 if torch.cuda.is_available() else 0)
            s = ('%10s'*2 + '%10.4g') % ('%g/%g' % (epoch, epochs - 1), mem, mean_loss)
            s += string_metrics
            pbar.set_description(s)
            
        scheduler.step()
        
        with torch.no_grad():
            val_metrics = validate(model, valloader, hyp, num_unfreezed_blocks=num_blocks)
            print_metrics(
                val_metrics['loss'],
                val_metrics['accuracy_mean'],
                val_metrics['f1_mean'],
                val_metrics['precisions'], 
                val_metrics['recalls'], 
                val_metrics['num_samples']
            )
            results_file = weights_save_dir.parent / 'results.txt'
            write_validation_metrics(results_file,
                epoch, 
                val_metrics['loss'],
                val_metrics['accuracy_mean'], 
                val_metrics['f1_mean'],
                val_metrics['precisions'],
                val_metrics['recalls']
            )
            write_metrics2tensorboard(tb_writer,
                train_metrics, 
                val_metrics,
                mean_loss, 
                epoch,
                num_blocks, 
                int(cnfg["epochs"])
            )
        
        # Saving model 
        if val_metrics['loss'] < best_val_loss:
            torch.save(model.state_dict(), best_path)
            best_val_loss = val_metrics['loss']
        if Path(prev_last).is_file():
            Path(prev_last).unlink()
        torch.save(model.state_dict(), last_path)
    
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnfg', type=str, default='./config.yaml')
    parser.add_argument('--hyp', type=str, default='./dataloader/hyp.yaml')
    args = parser.parse_args()

    cnfg = load_yaml(Path(args.cnfg))
    hyp = load_yaml(Path(args.hyp))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_dir, tensorboard_dir = create_runs_structure()
    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
    
    write_train_configuration((Path(args.cnfg), Path(args.hyp)), weights_dir.parent)
    print(f'Hyperparameters: {hyp}\n')
    print(f'Settings: {cnfg}\n')

    train(hyp, cnfg, device, weights_dir, tb_writer)
    return None

if __name__ == '__main__':
    main()