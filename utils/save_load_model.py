import torch

def save_checkpoint(filepath,model,optimizer,loss,accuracy,epoch):
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'loss':loss,
                'accuracy':accuracy,
                'epoch':epoch},filepath)
    
def load_checkpoint(filepath,model,optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    return model,optimizer,loss,accuracy,epoch
