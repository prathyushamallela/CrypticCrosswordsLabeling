from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.adapters import T5AdapterModel
import torch

class T5model(torch.nn.modules):

    def __init__():
        super.__init__()
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

class BiLSTMClassifier(torch.nn.modules):

    def __init__(self,input_dim, output_dim, hidden_dim = 512):
        super.__init__()
        self.bilstm = torch.nn.LSTM(input_dim,hidden_dim,bidirectional=True)
        self.linear = torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        output, (ht,ct) = self.bilstm(x)
        output = self.linear(x)
        return output