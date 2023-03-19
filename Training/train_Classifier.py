import sys
import pathlib
sys.path.insert(0,str(pathlib.Path().cwd()))

from Models.T5model import BiLSTMClassifier
from config.configuration import T5_type,clue_type_classes, save_file_path, dev
from utils.save_load_model import load_checkpoint,save_checkpoint
import torch
from DataLoader.load_DataLoader_classifier import train_dataloader, val_dataloader

## Initialization
classifier = BiLSTMClassifier(T5_type,len(clue_type_classes)).to(dev)
parameters = filter(lambda p: p.requires_grad, classifier.parameters())
lr = 0.001
optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.0001)
loss = 0
cur_epoch = 0
epoch = 10
criterion = torch.nn.CrossEntropyLoss()
accuracy = 0

## Load checkpoint if there is one
filepath = save_file_path / 'classifier.pt'
if filepath.exists():
    classifier, optimizer,loss, accuracy,cur_epoch = load_checkpoint(filepath,classifier,None)

for i in range(cur_epoch,epoch):
    ## Training Section
    classifier.train()
    train_correct = 0
    train_total = 0
    train_sum_loss = 0.0
    for x,y in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = classifier(torch.Tensor(x))
        loss = criterion(y_pred,y)
        train_total += len(y)
        train_correct += y_pred.eq(y).sum()
        train_sum_loss += loss.item()
        loss.backward()
        optimizer.step()

    ## Validation Section
    classifier.eval()
    val_correct = 0
    val_total = 0
    val_sum_loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for x,y in enumerate(val_dataloader):
        y_pred = classifier(x)
        loss = crit(y_pred, y)
        val_total += len(y)
        val_correct += y_pred.eq(y).sum()
        val_sum_loss += loss.item()


    if accuracy< val_correct/val_total:
        save_checkpoint(filepath,classifier,optimizer,loss,val_correct/val_total,i)
    print("epoch %d train loss %.3f, train acc %.3f, val loss %.3f, val accuracy %.3f" % (i, train_sum_loss/train_total, train_correct/train_total, val_sum_loss/val_total, val_correct/val_total))
    