import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import numpy as np
import evaluate
from config import get_config
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import save_checkpoint, get_grad_norm
import argparse
import random
import time
import datetime
import logging

# Utility classes from classification directory
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_tensor(tensor):
    rt = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
    return rt

def create_logger(output_dir, dist_rank, name):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = logging.Formatter(fmt)

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(color_fmt)
        logger.addHandler(console_handler)

        # create file handlers
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

    return logger

# Correct implementations matching the original
class RepBN(nn.Module):
    def __init__(self, channels):
        super(RepBN, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))  # Single parameter, not per-channel
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x) + self.alpha * x
        x = x.transpose(1, 2)
        return x

class LinearNorm(nn.Module):
    def __init__(self, dim, norm1=nn.LayerNorm, norm2=RepBN, warm=0, step=300000, r0=1.0):
        super(LinearNorm, self).__init__()
        self.register_buffer('warm', torch.tensor(warm))
        self.register_buffer('iter', torch.tensor(step))
        self.register_buffer('total_step', torch.tensor(step))
        self.r0 = r0
        self.norm1 = norm1(dim)
        self.norm2 = norm2(dim)

    def forward(self, x):
        if self.training:
            if self.warm > 0:
                self.warm.copy_(self.warm - 1)
                x = self.norm1(x)
            else:
                lamda = self.r0 * self.iter / self.total_step
                if self.iter > 0:
                    self.iter.copy_(self.iter - 1)
                x1 = self.norm1(x)
                x2 = self.norm2(x)
                x = lamda * x1 + (1 - lamda) * x2
        else:
            x = self.norm2(x)
        return x

def replace_layernorm_with_linearnorm(model, config):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            hidden_size = module.normalized_shape[0]
            setattr(model, name, LinearNorm(hidden_size, 
                                           warm=config.MODEL.TRANSITION_WARMUP_STEPS, 
                                           step=config.MODEL.TRANSITION_TOTAL_STEPS, 
                                           r0=config.MODEL.TRANSITION_R0))
        else:
            replace_layernorm_with_linearnorm(module, config)

def main():
    config = get_config()
    parser = argparse.ArgumentParser(description="BERT Gradual Training")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    args, unparsed = parser.parse_known_args()
    
    # Update config with args
    config = get_config(args)
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    # Create output directories
    os.makedirs(config.TRAIN.OUTPUT_DIR_STSB, exist_ok=True)
    os.makedirs(config.TRAIN.OUTPUT_DIR_RTE, exist_ok=True)
    
    # Create loggers
    stsb_logger = create_logger(config.TRAIN.OUTPUT_DIR_STSB, rank, "STSB")
    rte_logger = create_logger(config.TRAIN.OUTPUT_DIR_RTE, rank, "RTE")

    if rank == 0:
        print("Using device:", device)
        stsb_logger.info(f"Config:\n{config.dump()}")

    best_stsb_score = train_task('stsb', config, device, local_rank, stsb_logger)
    best_rte_score = train_task('rte', config, device, local_rank, rte_logger)

    if rank == 0:
        print("--- Final Results ---")
        print(f"Best STS-B Pearson Score: {best_stsb_score:.4f}")
        print(f"Best RTE Accuracy: {best_rte_score:.4f}")

def train_task(task_name, config, device, local_rank, logger):
    rank = dist.get_rank() if local_rank != -1 else 0
    world_size = dist.get_world_size() if local_rank != -1 else 1
    
    if rank == 0:
        logger.info(f"--- Training for {task_name.upper()} ---")

    tokenizer = BertTokenizer.from_pretrained(config.MODEL.NAME)
    dataset = load_dataset('glue', task_name)
    num_labels = config.MODEL.NUM_CLASSES_STSB if task_name == 'stsb' else config.MODEL.NUM_CLASSES_RTE
    
    def preprocess_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], 
                        truncation=True, padding='max_length', max_length=config.DATA.MAX_LENGTH)
    
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    columns = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in encoded_dataset['train'].features:
        columns.append('token_type_ids')
    encoded_dataset.set_format(type='torch', columns=columns)

    train_dataset = encoded_dataset['train']
    eval_dataset = encoded_dataset['validation']

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank) if local_rank != -1 else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.DATA.BATCH_SIZE, sampler=train_sampler, 
        num_workers=config.DATA.NUM_WORKERS, pin_memory=True, shuffle=(train_sampler is None))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.DATA.BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(config.MODEL.NAME, num_labels=num_labels)
    replace_layernorm_with_linearnorm(model, config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"Number of params: {n_parameters}")

    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer = build_optimizer(config, model)
    epochs = config.TRAIN.EPOCHS_STSB if task_name == 'stsb' else config.TRAIN.EPOCHS_RTE
    n_iter_per_epoch = len(train_loader)
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch, epochs)
    
    if rank == 0:
        logger.info("Start Training")
    
    best_score = 0.0

    start_time = time.time()
    for epoch in range(epochs):
        if local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        train_one_epoch(config, model, train_loader, optimizer, lr_scheduler, device, epoch, epochs, rank, n_iter_per_epoch, logger)
        score = validate(model, eval_loader, device, task_name, epoch, rank, logger)
        if rank == 0 and score > best_score:
            best_score = score
            logger.info(f'New best score: {best_score:.4f}')
            
            # Save best checkpoint
            if rank == 0:
                output_dir = config.TRAIN.OUTPUT_DIR_STSB if task_name == 'stsb' else config.TRAIN.OUTPUT_DIR_RTE
                save_state = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_score': best_score,
                    'epoch': epoch,
                    'config': config,
                    'task': task_name
                }
                save_path = os.path.join(output_dir, f'best_{task_name}_model.pth')
                torch.save(save_state, save_path)
                logger.info(f"Best model saved to {save_path}")

    if rank == 0:
        logger.info("Start BatchNorm Recalibration")
    
    # BatchNorm recalibration phase - exactly like classification/main.py
    for e in range(config.TRAIN.RECALIBRATION_EPOCHS):
        if local_rank != -1:
            train_loader.sampler.set_epoch(e + epochs)
        update_model(config, train_loader, model, logger, rank, device)
        score = validate(model, eval_loader, device, task_name, e, rank, logger, prefix="Recalibration")
        if rank == 0 and score > best_score:
            best_score = score
            logger.info(f'New best score after recalibration: {best_score:.4f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        logger.info('Training time {}'.format(total_time_str))
    
    return best_score

def train_one_epoch(config, model, data_loader, optimizer, lr_scheduler, device, epoch, total_epochs, rank, n_iter_per_epoch, logger):
    model.train()
    scaler = GradScaler(enabled=config.AMP)
    
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    start_time = time.time()
    end = time.time()
    
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16, enabled=config.AMP):
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        if config.TRAIN.CLIP_GRAD:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step_update(epoch * n_iter_per_epoch + i)

        # Update meters
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), labels.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and (i + 1) % config.TRAIN.LOGGING_STEPS == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (len(data_loader) - i)
            logger.info(
                f'Train: [{epoch + 1}/{total_epochs}][{i + 1}/{len(data_loader)}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    
    epoch_time = time.time() - start_time
    if rank == 0:
        logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(model, data_loader, device, task_name, epoch, rank, logger, prefix="Validation"):
    model.eval()
    metric = evaluate.load("glue", task_name)
    
    batch_time = AverageMeter()
    end = time.time()
    
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        outputs = model(input_ids, attention_mask=attention_mask)
        
        predictions = outputs.logits
        if task_name == 'stsb':
            predictions = predictions.squeeze()
        else:
            predictions = torch.argmax(predictions, dim=-1)

        metric.add_batch(predictions=predictions, references=labels)
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if rank == 0 and (i + 1) % 50 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'{prefix}: [{i + 1}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    eval_metric = metric.compute()
    if rank == 0:
        logger.info(f"{prefix} Epoch {epoch+1} evaluation - {eval_metric}")
    
    if task_name == 'stsb':
        return eval_metric['pearson']
    else:
        return eval_metric['accuracy']

@torch.no_grad()
def update_model(config, data_loader, model, logger, rank, device):
    """BatchNorm recalibration - exactly like classification/main.py"""
    model.eval()
    
    # Set BatchNorm layers to training mode but freeze their parameters
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    
    batch_time = AverageMeter()
    end = time.time()
    
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        # Forward pass to update BatchNorm statistics
        model(input_ids, attention_mask=attention_mask)
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if rank == 0 and (i + 1) % 100 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Recalibrating: [{i + 1}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

if __name__ == '__main__':
    main() 