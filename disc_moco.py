import warnings
warnings.filterwarnings('ignore')


import data, models_moco, utils
import argparse, os, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from sscd.datasets.disc import DISCTrainDataset, DISCEvalDataset

from tqdm import tqdm
import numpy as np


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--port-num',      default='9998', type=str)
parser.add_argument('--world-size',    default=2, type=int, help='number of gpus for ddp')

parser.add_argument('--data-dir',      default='/nfs_shared/Imagenet', type=str)
parser.add_argument('--batch-size',    default=256, type=int)
parser.add_argument('--num-workers',   default=16, type=int)

parser.add_argument('--moco-dim',      default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k',        default=65536, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m',        default=0.999, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t',        default=0.2, type=float, help='softmax temperature')

parser.add_argument('--start-epoch',   default=0, type=int)
parser.add_argument('--epochs',        default=200, type=int)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--min-lr',        default=1e-8, type=float)

parser.add_argument('--lr',            default=0.03, type=float)
parser.add_argument('--momentum',      default=0.9, type=float)
parser.add_argument('--weight-decay',  default=1e-4, type=float)

parser.add_argument('--resume',        default='', type=str, help='latest checkpoint')
parser.add_argument('--save',          action='store_true', help='save logs, checkpoints')
parser.add_argument('--save-name',     default='MoCo_DISC_200', type=str)
parser.add_argument('--save-freq',     default=1, type=int)
parser.add_argument('--print-freq',    default=200, type=int)
parser.add_argument('--log',           default='./logs/', type=str)
parser.add_argument('--checkpoint',    default='./checkpoints/', type=str)
args = parser.parse_args()



def init_process(gpu, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(gpu)
    dist.init_process_group('nccl', world_size=world_size, rank=gpu)

def main(gpu, world_size):
    init_process(gpu, world_size)

    # dataloader
    # train_dataset = data.PretrainDB(os.path.join(args.data_dir, 'train'))
    train_dataset = DISCTrainDataset(os.path.join(args.data_dir, 'train'))
    train_sampler = DistributedSampler(train_dataset, rank=gpu, num_replicas=args.world_size, shuffle=True, drop_last=True)
    train_loader  = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=int(args.batch_size // args.world_size),
                                                shuffle=False,
                                                sampler=train_sampler,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=True)
    if gpu == 0:                                            
        print(f"train lengths: {len(train_dataset)}")

    # val_dataset = data.PretrainDB(os.path.join(args.data_dir, 'val'))
    val_dataset = DISCEvalDataset("/nfs_shared/DISC/val_images")
    val_sampler = DistributedSampler(val_dataset, rank=gpu, num_replicas=args.world_size, shuffle=True, drop_last=True)
    val_loader  = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=int(args.batch_size // args.world_size),
                                              shuffle=False,
                                              sampler=val_sampler,
                                              num_workers=args.num_workers,
                                            #   pin_memory=True,
                                              persistent_workers=True)

    # model
    net = models_moco.MoCo(args.moco_dim, args.moco_k, args.moco_m, args.moco_t).cuda(gpu)
    net = DistributedDataParallel(net, device_ids=[gpu])

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # critertion
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # logger
    logger = utils.Logger(args)
    if dist.get_rank() == 0: logger.initialize()

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(gpu))
        args.start_epoch = checkpoint['epoch']
        net.module.encoder_q.load_state_dict(checkpoint['q_state_dict'])
        net.module.encoder_k.load_state_dict(checkpoint['k_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # epoch start
    best_map = 0
    for epoch in range(args.start_epoch, args.epochs):
        # train
        net.train()
        if dist.get_rank() == 0: print('Epoch {} Train Started...'.format(epoch))

        train_loss = []
        train_start = time.time()
        for i, (imgs_q, imgs_k) in enumerate(train_loader):
            lr = utils.cosine_scheduler(optimizer, epoch + i/len(train_loader), args)

            imgs_q, imgs_k = imgs_q.cuda(gpu), imgs_k.cuda(gpu)
            output, target = net(imgs_q, imgs_k)
            loss = criterion(output, target)

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0: train_loss.append(loss.item() / args.world_size)

            if (i % args.print_freq == 0) and (dist.get_rank() == 0):
                print('Iteration : {:0>5}   LR : {:.6f}   Train Loss : {:.6f}'.format(i, lr, train_loss[-1]))

        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start))
        print(f"train elapsed time: {train_time}")

        # val
        net.eval()
        if dist.get_rank() == 0: print('Epoch {} Val Started...'.format(epoch))

        val_start = time.time()
        with torch.no_grad():
            outputs = []
            if gpu == 0:
                data_iter = tqdm(val_loader, desc="GPU 0")
            else:
                data_iter = val_loader

            for batch in data_iter:
                input = batch["input"].cuda(gpu)
                metadata_keys = ["image_num", "split", "instance_id"]
                batch = {k: v for (k, v) in batch.items() if k in metadata_keys}
                batch["embeddings"] = net(input, eval=True)
                outputs.append(batch)

        dist.barrier()
        keys = ["embeddings", "image_num", "split", "instance_id"]
        outputs = {k: torch.cat([out[k] for out in outputs]) for k in keys}
        outputs = _gather(outputs)

        if dist.get_rank() == 0:
            outputs = {key: tensor.to('cpu') for key, tensor in outputs.items()}
            outputs = dedup_outputs(outputs)
            if epoch == 0:
                print(
                    "Eval dataset size: %d (%d queries, %d index)"
                    % (
                        outputs["split"].shape[0],
                        (outputs["split"] == DISCEvalDataset.SPLIT_QUERY).sum(),
                        (outputs["split"] == DISCEvalDataset.SPLIT_REF).sum(),
                    )
                )
            metrics = val_dataset.retrieval_eval(
                outputs["embeddings"],
                outputs["image_num"],
                outputs["split"],
            )
            metrics = {k: 0.0 if v is None else v for (k, v) in metrics.items()}

            if metrics['uAP'] > best_map:
                path = os.path.join(args.log, f"best_epoch_{epoch}_{metrics['uAP']}.ckpt")
                torch.save(net, path)

                best_map = metrics['uAP']
            metrics_str = ', '.join([f"{k}: {round(v, 3)}" for k, v in metrics.items()])
            print(metrics_str)
        
            val_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - val_start))
            print(f"train elapsed time: {val_time}")

            # save checkpoint
            if args.save and (epoch % args.save_freq == 0):
                checkpoint = os.path.join(args.checkpoint, '{}_{:0>4}.pth.tar'.format(args.save_name, epoch))
                torch.save({'epoch' : epoch+1,
                            'q_state_dict' : net.module.encoder_q.state_dict(),
                            'k_state_dict' : net.module.encoder_k.state_dict(),
                            'optimizer' : optimizer.state_dict()},
                             checkpoint)

def _gather(outputs):
    gathered_outputs = {key: None for key in outputs.keys()}
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for key in outputs.keys():
        output = outputs[key].cuda()
        if output.is_sparse:
            output = output.to_dense()
        output_list = [torch.zeros_like(output) for _ in range(world_size)] if rank == 0 else None
        dist.gather(output, gather_list=output_list, dst=0)
        if rank == 0:
            gathered_outputs[key] = torch.cat(output_list, dim=0)

    return gathered_outputs

def dedup_outputs(outputs, key="instance_id"):
    """Deduplicate dataset on instance_id."""
    idx = np.unique(outputs[key].numpy(), return_index=True)[1]
    outputs = {k: v.numpy()[idx] for (k, v) in outputs.items()}
    assert np.unique(outputs[key]).size == outputs["instance_id"].size
    return outputs

def run(world_size):
    torch.multiprocessing.spawn(main, nprocs=world_size, args=(world_size,))
    dist.destroy_process_group()


if __name__ == '__main__':
    print('Available GPUs : {}   Use GPUs : {}'.format(torch.cuda.device_count(), args.world_size))
    assert args.world_size <= torch.cuda.device_count()
    run(args.world_size)
