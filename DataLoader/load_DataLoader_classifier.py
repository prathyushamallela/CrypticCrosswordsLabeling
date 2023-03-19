from torch.utils.data import  DataLoader
from datasets import load_dataset
from config.configuration import data_file_path, data_file_name, batch_size

    
train_dataloader = DataLoader(load_dataset("json",data_files = str(data_file_path/data_file_name) )['train'],batch_size= batch_size, shuffle= True)