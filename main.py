import os
import time
import torch
import tarfile
import shutil
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
resume = True

num_features = 1408
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
        self.features._fc = nn.Linear(self.num_features,self.num_classes)
        # self.features._fc = nn.Sequential(nn.Linear(self.num_features, 56),
        #                          nn.BatchNorm1d(56),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(56, self.num_classes))


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
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        # log_softmax = torch.nn.LogSoftmax(dim=1)
        # output =  log_softmax(output )
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
                             prefix='Val: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            # log_softmax = torch.nn.LogSoftmax(dim=1)
            # output =  log_softmax(output )
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.print(i)

    return top1.avg

def save_checkpoint(state,  filename='checkpoint.pth'):
    torch.save(state, filename)
    
    shutil.copyfile(filename, 'model_best.pth')

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

    # model = Pollen(20,num_features).to(device)
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Sequential(nn.Linear(features_[model_name], 20),nn.LogSoftmax(dim=1))
                          
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if resume:
        checkpoint = torch.load('checkpoint.pth')
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        
            # best_acc1 may be from a checkpoint from a different GPU
        best_acc1 = best_acc1.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('MODEL RESUMEEEEDD')
            
    # print(model)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    verbose=True,
    factor=0.2
    )

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = torch.nn.NLLLoss().to(device)
    best_val_accuracy = 0
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, device)
        val_acc = vaidate(val_loader, model, criterion, device)
        scheduler.step(val_acc)
        
        if val_acc > best_val_accuracy:
            torch.save(model,save_filename_f)
            best_val_accuracy = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_val_accuracy,
                'optimizer' : optimizer.state_dict(),
            })
        # print(train_loss,train_acc )

        # print( val_loss, val_acc)
        # exit()

if __name__ == '__main__':
    main()