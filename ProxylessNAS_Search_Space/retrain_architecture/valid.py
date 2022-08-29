import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import argparse
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from thop import profile
import pdb
from model import Network
import numpy as np
import pickle
from config import config

IMAGENET_TEST_SET_SIZE = 50000

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--batch_size', type=int, default=64, help='batch size') #1024
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=240, help='num of training epochs')
parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
parser.add_argument('--checkpoint', type=str, default='./eval_retrained_model', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--model_id', type=str, default='1 1 -1 2 1 -1 5 1 1 -1 5 1 1 -1 1 4 3 -1 5 5 5', help='model_id') # '1 1 -1 2 1 -1 5 1 1 -1 5 1 1 -1 1 4 3 -1 5 5 5'
parser.add_argument('--data', metavar='DIR', default='/home/public/imagenet/', help='path to dataset')
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')

args = parser.parse_args()

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def get_arch_flops(op_flops_dict, cand):
    assert len(cand) == len(config.backbone_info) - 2
    preprocessing_flops = op_flops_dict['PreProcessing'][config.backbone_info[0]]
    postprocessing_flops = op_flops_dict['PostProcessing'][config.backbone_info[-1]]
    total_flops = preprocessing_flops + postprocessing_flops
    for i in range(len(cand)):
        inp, oup, img_h, img_w, stride = config.backbone_info[i+1]
        op_id = cand[i]
        if op_id >= 0:
          key = config.blocks_keys[op_id]
          total_flops += op_flops_dict[key][(inp, oup, img_h, img_w, stride)]
    return total_flops

def main():
  if not torch.cuda.is_available():
    print('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.deterministic = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  print('gpu device = %d' % args.gpu)

  # The network architeture coding
  rngs = [int(id) for id in args.model_id.split(' ')]
  model = Network(rngs)
  op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))
  flops = get_arch_flops(op_flops_dict, rngs)
  params = utils.count_parameters_in_MB(model)
  model = model.cuda(args.gpu)

  arch = model.architecture()

  print('flops = {}MB, param size = {}MB'.format(flops/1e6, params))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  # Data loading code
  valdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          normalize,
      ])),
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)
  

  best_acc_top1 = 0
  checkpoint_tar = os.path.join(args.checkpoint, 'att_search_checkpoint.pth.tar')
  if os.path.exists(checkpoint_tar):
      print('loading checkpoint {} ..........'.format(checkpoint_tar))
      checkpoint = torch.load(checkpoint_tar)
      best_acc_top1 = checkpoint['best_acc_top1']
      print('best_acc_top1 = ', best_acc_top1)

      model_weights_no_module = {}
      for key in checkpoint['state_dict']:
          if 'module' in key:
              weights_name = key[7:]
          else:
              weights_name = key
          model_weights_no_module[weights_name] = checkpoint['state_dict'][key]
      model.load_state_dict(model_weights_no_module)
      model.eval()

      print("loaded checkpoint {}" .format(checkpoint_tar))

  valid_acc_top1, valid_acc_top5, valid_obj = infer(val_loader, model, criterion)
  print('valid_acc_top1: {} valid_acc_top5: {} best_acc_top1: {}'.format(valid_acc_top1, valid_acc_top5, best_acc_top1))


def infer(val_loader, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for i, (image, target) in enumerate(val_loader):

      image = image.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

      logits = model(image)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = image.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if i % args.report_freq == 0:
        print('valid %03d %e %f %f' % (i, objs.avg, top1.avg, top5.avg))
  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 
