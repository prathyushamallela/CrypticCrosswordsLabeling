from torch.utils.data import  DataLoader
from datasets import load_dataset
from config.configuration import data_file_path, data_file_name, batch_size


train_ds = load_dataset("json",data_files = str(data_file_path/data_file_name) , split = ['train[:80%]'])
test_ds = load_dataset("json",data_files = str(data_file_path/data_file_name) , split = ['train[80%:]'])

train_dataloader = DataLoader(train_ds,batch_size= batch_size, shuffle= True)