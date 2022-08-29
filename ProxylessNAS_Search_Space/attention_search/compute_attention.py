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
import pickle
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
parser.add_argument('--pretrained_path', type=str, default='./exp_log/checkpoint.pth.tar', help='save path')

args = parser.parse_args()

save_path = '{}/attention_eval-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

    # Prepare model
    model = AttSuperNetwork().cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    # load pretrained weights
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # pretrained_path = './exp_log/checkpoint_epoch_120.pth.tar'
    pretrained_weights = torch.load(args.pretrained_path, map_location=device)
    model.load_state_dict(pretrained_weights['state_dict'])

    # load validation dataset
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = datasets.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    valid_acc, valid_obj = infer(valid_loader, model, criterion)
    logging.info('valid_acc %f valid_loss %f' % (valid_acc, valid_obj))
    print('\n')


def handle_attention_weights(attention_weights, his_attention_weights=None):
    new_attention_weights = []
    for idx, _attention_weights in enumerate(attention_weights):
        channel_dim = _attention_weights[0].shape[1]
        op_weights = [torch.sum(_weights).item()/channel_dim for _weights in _attention_weights]
        if his_attention_weights is not None:
            op_weights = np.array(op_weights) + his_attention_weights[idx]
        else:
            op_weights = np.array(op_weights)
        new_attention_weights.append(op_weights)
    return new_attention_weights


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

            logits, attention_weights = model(input, return_attention=True)
            if step == 0:
                his_attention_weights = handle_attention_weights(attention_weights, his_attention_weights=None)
            else:
                his_attention_weights = handle_attention_weights(attention_weights, his_attention_weights=his_attention_weights)
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info(his_attention_weights)
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    with open('attention_weights.pickle', 'wb') as f:
        pickle.dump(his_attention_weights, f)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
