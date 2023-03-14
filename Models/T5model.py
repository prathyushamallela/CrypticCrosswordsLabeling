from config.Classes import clue_type_classes
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import T5AdapterModel, PrefixTuningConfig 
import torch

class pre_trained_T5Tokenizer():

    def __init__(self,t5_type):
        self.T5tokenizer = T5Tokenizer.from_pretrained(t5_type)

    ## 'Input' is list of string
    def tokenize(self,input):
        if type(input) == str:
            input = [input] 
        return self.T5tokenizer(input,return_tensors = 'pt',padding = True).input_ids
    
    def decode(self,input):
        return self.T5tokenizer.batch_decode(input,skip_special_tokens=True)

class BiLSTMClassifier(torch.nn.Module):

    def __init__(self,t5_type, output_dim, hidden_dim = 512):
        super().__init__()
        self.embedding = T5ForConditionalGeneration.from_pretrained(t5_type).get_input_embeddings()
        self.bilstm = torch.nn.LSTM(T5Config.from_pretrained(t5_type).d_model,hidden_dim,bidirectional=True,batch_first = True)
        self.linear = torch.nn.Linear(hidden_dim * 2,output_dim)

    ## Expect dimension of x = batch x sequences 
    def forward(self,x):
        output = self.embedding(x)
        output, (ht,ct) = self.bilstm(output)
        output = self.linear(ht.permute(1,0,2).reshape(ht.shape[1],-1))
        output = torch.nn.functional.softmax(output,dim = 1)
        output = torch.argmax(output,dim = 1, keepdim = True)
        return output
    
class T5model(torch.nn.Module):

    def __init__(self,t5_type,classes):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(t5_type)
        ## Freeze T5 model
        for p in self.model.parameters():
            p.requires_grad = False
        for i in classes:
            self.add_adapter(i,PrefixTuningConfig(flat = False, prefix_length = 30))

    def forward(self,x,clue_type):
        self.activate_adapter(clue_type)
        output = self.model.generate(x)
        return output
    
    def add_adapter(self,task_name,config):
        self.model.add_adapter(task_name,config)

    def activate_adapter(self,task_name):
        self.model.set_active_adapters(task_name)
    
class CrypticCrosswordSolver(torch.nn.Module):

    def __init__(self,T5_type, classes,classifier):
        super().__init__()
        self.tokenizer = pre_trained_T5Tokenizer(T5_type)
        self.classifier =  classifier
        self.t5 = T5model(T5_type,classes)
        self.classes = classes

    def forward(self,x):
        output = self.tokenizer.tokenize(x)
        clue_type = self.classifier(output)
        idx =  clue_type.squeeze().tolist()[0]
        clue_type = self.classes[idx]
        output = self.t5(output,clue_type)
        output = self.tokenizer.decode(output)
        return output

T5_type = "t5-small"

mock_input = ["This is a mock data, I am testing.","Another mock data sentence, also for testing","Another one for testing"]

solver = CrypticCrosswordSolver(T5_type,clue_type_classes,BiLSTMClassifier(T5_type,len(clue_type_classes)))
output = solver(mock_input)
