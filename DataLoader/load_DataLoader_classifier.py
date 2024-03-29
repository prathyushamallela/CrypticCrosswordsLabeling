from datasets import load_dataset
from config.configuration import data_file_path, batch_size
from torch.utils.data import  DataLoader
from utils.preprocess_data import tokenize_for_classifier

train_ds = load_dataset("json",data_files = str(data_file_path/"trainsample1_1679513775.707641.jsonl") , split = 'train[:80%]')
val_ds = load_dataset("json",data_files = str(data_file_path/"trainsample1_1679513775.707641.jsonl") , split = 'train[80%:]')

train_ds = train_ds.map(tokenize_for_classifier,batched= True)
val_ds = val_ds.map(tokenize_for_classifier,batched= True)

train_dataloader = DataLoader(train_ds,batch_size= batch_size, shuffle= True)
val_dataloader = DataLoader(val_ds,batch_size= batch_size, shuffle= True)