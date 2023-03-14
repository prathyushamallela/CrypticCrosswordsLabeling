from config.Classes import clue_type_classes
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import T5AdapterModel 
import torch

class pre_trained_T5Tokenizer():

    def __init__(self,t5_type):
        self.T5tokenizer = T5Tokenizer.from_pretrained(t5_type)

    ## 'Input' is list of string
    def tokenize(self,input):
        if type(input) == str:
            input = [input] 
        return self.T5tokenizer(input,return_tensors = 'pt',padding = True).input_ids

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
    
class pre_trained_T5model(torch.nn.Module):

    def __init__(self,t5_type):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(t5_type)

class T5Adapter():

    def __init__(self,T5model):
        self.T5model = T5model.model
        self.T5model.add_adapter('test')
    
class CrypticCrosswordSolver():

    def __init__(self,classifier):
        self.classifier = classifier

T5_type = "t5-small"

mock_input = ["This is a mock data, I am testing.","Another mock data sentence, also for testing","Another one for testing"]

tokenizer = pre_trained_T5Tokenizer(T5_type)
classifer = BiLSTMClassifier(T5_type,len(clue_type_classes))

input = tokenizer.tokenize(mock_input)
output = classifer(input)
print(output.shape)