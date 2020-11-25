import os
import time
import torch
import tarfile
import torchvision
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from config_file import *
from efficientnet_pytorch import EfficientNet

from utils import AverageMeter, ProgressMeter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


def accuracy(outputs,labels):
    _, preds = torch.max(outputs,dim=1)
    return torch.tensor(torch.sum(preds ==labels).item() / len(preds))


class Pollen(nn.Module):
    def __init__(self, num_classes, num_features):
        super(Pollen, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.features = EfficientNet.from_pretrained(model_name)
        self.features._fc = nn.Sequential(nn.Linear(self.num_features, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(128, 56),
                                 nn.BatchNorm1d(56),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(56, self.num_classes))


    def forward(self, img_data):
        output = self.features(img_data)

        return output


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time,  losses, top1,
                              prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)


def vaidate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    # Dataset
    data_path = 'data_split'
    train_ds = ImageFolder(data_path+'/train', transform=transform_train)
    val_ds = ImageFolder(data_path+'/val', transform=transform_test)
    pred_ds = ImageFolder(data_path+'/test', transform=transform_test)

    #data_loader

    train_loader = DataLoader(train_ds, train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, test_val_batch_size , shuffle=False, num_workers=4, pin_memory=True)
    pred_loader = DataLoader(pred_ds, test_val_batch_size , shuffle= False, num_workers=4, pin_memory=True)

    model = Pollen(20,1280).to(device)
    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=5,
    verbose=True,
    factor=0.2
    )

    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, device)
        val_loss = vaidate(val_loader, model, criterion, device)

        # print(train_loss,train_acc )

        # print( val_loss, val_acc)
        # exit()

if __name__ == '__main__':
    main()