from utils import *
import argparse
import torch.nn as nn
import utils
import torch.backends.cudnn as cudnn

from att_supernet import AttSuperNetwork
from config import config
import functools
import torchvision.transforms as transforms

print = functools.partial(print, flush=True)
import time
import logging
from tensorboard_logger import configure

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.25, help='init learning rate')
parser.add_argument('--min_lr', type=float, default=5e-4, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
parser.add_argument('--classes', type=int, default=1000, help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--save', type=str, default='exp_log', help='save path')
parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
parser.add_argument('--data', metavar='DIR', default='../../../imagenet_sub', help='path to dataset')
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')

args = parser.parse_args()

save_path = '{}/search-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
configure("%s" % (save_path))

# IMAGENET_TRAINING_SET_SIZE = 260
IMAGENET_TRAINING_SET_SIZE = 128066
# IMAGENET_TRAINING_SET_SIZE = 1281167
train_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size
args.total_iters = train_iters * args.epochs


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    if args.local_rank == 0:
        logging.info("args = %s", args)

    group_name = 'spos_random_label_supernet_training'
    torch.distributed.init_process_group(backend='nccl', init_method='env://', group_name=group_name)
    args.world_size = torch.distributed.get_world_size()
    args.distributed = args.world_size > 1
    args.batch_size = args.batch_size // args.world_size
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(args.classes, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # Prepare model
    model = AttSuperNetwork().cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)
    # model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    optimizer, scheduler = utils.get_optimizer_schedule(model, args)

    # # Prepare data
    # traindir = os.path.join(args.data, 'train')
    # train_transform = utils.get_train_transform()
    # train_dataset = utils.ImageNetWithRandomLabels(
    #     root=traindir,
    #     transform=train_transform
    # )
    #
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers//args.world_size, pin_memory=True, sampler=train_sampler)
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_dataset = datasets.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers // args.world_size, pin_memory=True, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)


    seed = args.seed
    start_epoch, ops_dim, modify_initial_model = 0, 0, False
    checkpoint_tar = os.path.join(config.exp_name, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar, map_location={'cuda:0': 'cuda:{}'.format(args.local_rank)})
        start_epoch = checkpoint['epoch'] + 1
        seed = checkpoint['seed']
        model.load_state_dict(checkpoint['state_dict'])

        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        logging.info('{} load checkpoint..., epoch = {}'.format(now, start_epoch))

        # reset the scheduler
        for _ in range(start_epoch):
            if scheduler.get_lr()[0] > args.min_lr:
                scheduler.step()
        logging.info('resume from epoch={}'.format(start_epoch))

    # Save the base weights for computing angle
    if start_epoch == 0 and args.local_rank == 0:
        utils.save_checkpoint({'epoch': 0,
                               'state_dict': model.state_dict(),
                               'seed': seed,
                               'optimizer': optimizer.state_dict(),
                               }, args.save)
        logging.info('save base weights ...')

    for epoch in range(start_epoch, args.epochs):
        # Supernet training
        train(train_loader, optimizer, scheduler, model, criterion_smooth, epoch, train_iters, args)

        if args.local_rank == 0:
            # Save the current model
            if (epoch + 1) % 5 == 0:
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'state_dict': model.state_dict(),
                                       'seed': seed,
                                       'optimizer': optimizer.state_dict(),
                                       }, args.save)
                valid_acc, valid_obj = infer(valid_loader, model, criterion)
                logging.info('valid_acc %f valid_loss %f' % (valid_acc, valid_obj))

        print('\n')


def train(train_loader, optimizer, scheduler, model, criterion, epoch, train_iters, args):
    objs, top1 = utils.AvgrageMeter(), utils.AvgrageMeter()
    model.train()
    for step, (image, target) in enumerate(train_loader):
        if scheduler.get_lr()[0] > args.min_lr:
            scheduler.step()
        t0 = time.time()
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        n = image.size(0)
        datatime = time.time() - t0

        logits = model(image)
        optimizer.zero_grad()
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0 and args.local_rank == 0:
            now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            logging.info('{} |=> Epoch={}, train:  {} / {}, loss={:.2f}, acc={:.2f}, lr={}, datatime={:.2f}' \
                         .format(now, epoch, step, train_iters, objs.avg, top1.avg, scheduler.get_lr()[0],
                                 float(datatime)))
    return objs.avg, top1.avg


def infer(valid_loader, model, criterion):
    logging.info("Start to Eval ...")

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
