from config.configuration import clue_type_classes,T5_type,adapter_config,mock_input
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from transformers.adapters import T5AdapterModel
import torch

## Vocabulary for class 
## Usage: from class to idx: idx = vocab.get_idx(class_name)
##        from idx to class: class_name = vocab.get_class(idx)
class Class_vocab():

    def __init__(self,classes):
        self.classes = classes
        self.class2idx = dict()
        self.idx2class = dict()
        count = 0
        for cl in classes:
            self.class2idx[cl] = count
            self.idx2class[count] = cl
            count += 1

    def get_idx(self,cl):
        return self.class2idx[cl]

    def get_class(self,idx):
        return self.idx2class[idx]
    
    def batch_get_idx(self,batch):
        return [ self.get_idx(cl) for cl in batch]

    def batch_get_class(self,batch):
        return [ self.get_class(idx.item()) for idx in batch]

## pre-trained T5 tokenizer
## Usage: from sentences to ids: tokens(id form) = tokenizer.tokenize(sentences)
##        from ids to sentences: sentences = tokenizer.decode(ids)
class pre_trained_T5Tokenizer():

    def __init__(self,t5_type):
        self.T5tokenizer = T5Tokenizer.from_pretrained(t5_type)

    ## 'Input' is list of string
    def tokenize(self,input):
        if type(input) == str:
            input = [input] 
        ids = self.T5tokenizer(input,return_tensors = 'pt',padding = True)
        return ids.input_ids, ids.attention_mask
    
    def decode(self,input):
        return self.T5tokenizer.batch_decode(input,skip_special_tokens=True)

## BiLSTM Classifier
## Initialization: classifier = BiLSTMClassifier(t5_type, number of class)
## Usage: predict_class_ids = classifier(sentences)
class BiLSTMClassifier(torch.nn.Module):

    def __init__(self,t5_type, output_dim,hidden_dim = 512, num_layers = 1):
        super().__init__()
        self.embedding = T5ForConditionalGeneration.from_pretrained(t5_type).get_input_embeddings()
        self.bilstm = torch.nn.LSTM(T5Config.from_pretrained(t5_type).d_model,hidden_dim,num_layers,bidirectional=True,batch_first = True)
        self.linear = torch.nn.Linear(hidden_dim * 2,output_dim)

    ## Expect dimension of x = batch x sequences 
    def forward(self,x):
        output = self.embedding(x)
        output, (ht,ct) = self.bilstm(output)
        output = self.linear(ht.permute(1,0,2).reshape(ht.shape[1],-1))
        output = torch.nn.functional.softmax(output,dim = 1)
        return output
    
class T5modelWithAdapter(torch.nn.Module):

    def __init__(self,t5_type,classes):
        super().__init__()
        self.model = T5AdapterModel.from_pretrained(t5_type)
        self.model.freeze_model(True)
        for i in classes:
            self.model.add_adapter(i,adapter_config)
            self.model.add_seq2seq_lm_head(i)

    def forward(self,x,attention_mask,clue_type,topk = 1):
        self.model.set_active_adapters(clue_type[0])
        ## For autoregressive inference instead of using decoder ids
        output = self.model.generate(x, attention_mask =attention_mask,num_beams = topk, num_return_sequences = topk)
        return output
    
class CrypticCrosswordSolver(torch.nn.Module):

    def __init__(self,T5_type, classes,classifier):
        super().__init__()
        self.tokenizer = pre_trained_T5Tokenizer(T5_type)
        self.classifier =  classifier
        self.t5 = T5modelWithAdapter(T5_type,classes)
        self.classes = Class_vocab(classes)

    def forward(self,x,topk = 1):
        output,attention_mask = self.tokenizer.tokenize(x)
        clue_type = self.classifier(output)
        clue_type  = torch.argmax(clue_type ,dim = 1, keepdim = True)
        clue_type = self.classes.batch_get_class(clue_type)
        output = self.t5(output,attention_mask,clue_type,topk = topk )
        output = self.tokenizer.decode(output)
        return output



# solver = CrypticCrosswordSolver(T5_type,clue_type_classes,BiLSTMClassifier(T5_type,len(clue_type_classes)))
# output = solver(mock_input,topk = 10)
# print(output)