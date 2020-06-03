import argparse
import os

import csv
from tqdm import tqdm
import torch
import time
import fusion_net
import video_net
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from RAVDESSDataset import RAVDESSDataset

model_options = ['fusion', 'fusion_VGG', 'video', 'video_VGG']
dataset_options = ['speech_mfccs', 'speech_audio', 'all_mfccs', 'all_audio', 'random_mfccs', 'random_audio', 'final_mfccs', 'final_audio']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='final_mfccs',
                    choices=dataset_options)
parser.add_argument('--model_type', '-a', default='fusion_VGG',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate')
parser.add_argument('--rho', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--eps', type=float, default=1e-8,
                     help='learning decay for lr scheduler')
parser.add_argument('--momentum', type=float, default=0.9,  metavar='M',
                    help='momentum')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--lr_decay', type=float, default=1e-4,
                     help='learning decay for lr scheduler')

def main():
    global args

    args = parser.parse_args()

    test_id = '20_05_27_' + str(args.dataset) + '_' + str(args.model_type) + '_' + str(args.batch_size)

    csv_path = 'logs/'
    model_path = 'checkpoints'

    print(test_id)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dataset = RAVDESSDataset(args.dataset, True)
    test_dataset = RAVDESSDataset(args.dataset, False)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model_type == 'fusion':
        model = fusion_net.fusion_net(args.dataset)
    elif args.model_type == 'fusion_VGG':
        model = fusion_net.fusion_net(args.dataset, True)
    elif args.model_type == 'video':
        model = video_net.video_net(args.dataset)
    elif args.model_type == 'video_VGG':
        model = video_net.video_net(args.dataset, True)

    model = model.cuda()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    #print('Trainable params num : ', sum(params_num))
    
    optimizer = optim.Adadelta(filtered_parameters, lr=args.lr, rho=args.rho, eps=args.eps)

    filename = csv_path + '/' + test_id + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'], filename=filename)

    criterion = nn.CrossEntropyLoss().cuda()

    best_acc = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)

        # train for one epoch
        model.train()
        train_acc, train_loss = train(progress_bar, model, criterion, optimizer, epoch)

        model.eval()
        test_acc, test_loss = test(test_loader, model, criterion)

        tqdm.write('train_loss: {0:.3f} train_acc: {1:.3f} / test_loss: {2:.3f} test_acc: {3:.3f}'.format(train_loss, train_acc, test_loss, test_acc))
        row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'train_acc': str(train_acc), 'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_type,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, model_path, test_id)
            tqdm.write('save_checkpoint')
        if epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_type,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, model_path, test_id+"_last")
    csv_logger.close()


def train(progress_bar, model, criterion, optimizer, epoch):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    losses = 0

    for i, (video, audio, target) in enumerate(progress_bar):
        video = video.cuda()
        audio = audio.cuda()
        target = torch.IntTensor(target)
        target = target.type(dtype=torch.long)
        target = target.cuda()

        if args.model_type.startswith('video'):
            output = model(video)
        else:
            output = model(audio, video)

        loss = criterion(output, target)

        progress_bar.set_description('Epoch : {0} / loss :{1:.3f}'.format(epoch, loss.item()))
        output = torch.max(output.data, 1)[1]

        total += target.size(0)
        correct += (output == target.data).sum().item()
        xentropy_loss_avg += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate running average of accuracy
    accuracy = correct / total
    train_loss = xentropy_loss_avg / i

    progress_bar.set_postfix(xentropy='%.3f' % (train_loss), acc='%.3f' % accuracy)

    return accuracy, train_loss

def test(loader, model, criterion):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    for i, (video, audio, target) in enumerate(loader):
        # measure data loading time
        video = video.cuda()
        audio = audio.cuda()
        target = torch.IntTensor(target)
        target = target.type(dtype=torch.long)
        target = target.cuda()
        
        with torch.no_grad():
            if args.model_type.startswith('video'):
                output = model(video)
            else:
                output = model(audio, video)

        loss = criterion(output, target)
        output = torch.max(output.data, 1)[1]
        total += target.size(0)

        correct += (output == target.data).sum().item()
        xentropy_loss_avg += loss.item()

    accuracy = correct / total
    test_loss = xentropy_loss_avg / i

    return accuracy, test_loss

class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def save_checkpoint(state, model_path, test_id):
    filename = model_path + '/' + test_id +'.pth.tar'
    torch.save(state, filename)

if __name__ == '__main__':
    main()
