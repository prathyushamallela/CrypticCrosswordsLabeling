import torch
from Models.T5model import CrypticCrosswordSolver, BiLSTMClassifier
from config.configuration import T5_type, clue_type_classes,save_file_path, dev, adapter_config

def save_checkpoint(filepath,model,optimizer,loss,accuracy,epoch):
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'loss':loss,
                'accuracy':accuracy,
                'epoch':epoch},filepath)
    
def load_checkpoint(filepath,model,optimizer):
    checkpoint = torch.load(filepath,map_location  = dev)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    return model,optimizer,loss,accuracy,epoch

def load_solver_model():
    model = CrypticCrosswordSolver(T5_type,clue_type_classes,BiLSTMClassifier(T5_type,len(clue_type_classes)).to(dev))
    checkpoint = torch.load(save_file_path/'classifier.pt',map_location  = dev)
    model.classifier.load_state_dict(checkpoint['model_state_dict'])
    checkpoint_folder = get_last_checkpoint_folder()
    model.t5.model.to(dev)
    for cl in clue_type_classes:
        model.t5.model.load_adapter(str(save_file_path/checkpoint_folder/cl),config = adapter_config)
    return model

def load_classifier():
    model = BiLSTMClassifier(T5_type,len(clue_type_classes)).to(dev)
    checkpoint = torch.load(save_file_path/'classifier.pt',map_location  = dev)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_last_checkpoint_folder():
    num = []
    for item in  save_file_path.iterdir():
        if item.is_dir() and 'checkpoint-' in item.name:
            num.append(int(item.name.replace('checkpoint-','')))
    return 'checkpoint-'+str(max(num))