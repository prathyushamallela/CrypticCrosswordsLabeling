from transformers import T5Tokenizer
from config.configuration import T5_type, clue_type_classes
from Models.T5model import Class_vocab


class data():

    def __init__(self,cluename,answer,length,anagram,container,reversal,deletion,homophone,charade,hidden,double,unclassified,predicted):
        self.cluename = cluename
        self.answer = answer
        self.length = length
        self.scores = dict()
        self.scores["anagram"] = anagram
        self.scores["container"] = container
        self.scores["reversal"] = reversal
        self.scores["deletion"] = deletion
        self.scores["homophone"] = homophone
        self.scores["charade"] = charade
        self.scores["hidden-word"] = hidden
        self.scores["double-def"] = double
        self.scores["unclassified"] = unclassified
        self.predicted_cluetype = predicted

    def get_clue(self):
        return self.cluename
    
    def get_answer(self):
        return self.answer
    
    def get_type(self):
        return self.predicted_cluetype
    
class json_data_class():
    
    def __init__(self,json_data):
        self.array = []
        for i in range(len(json_data["length"])):
            dat = data(json_data["cluename"][i],
                       json_data["answer"][i],
                       json_data["length"][i],
                       json_data["anagram_score"][i],
                       json_data["container_score"][i],
                       json_data["reversal_score"][i],
                       json_data["deletion_score"][i],
                       json_data["homophone_score"][i],
                       json_data["charade_score"][i],
                       json_data["hidden_word_score"][i],
                       json_data["double_definition_score"][i],
                       json_data["unclassified_score"][i],
                       json_data["predicted_cluetype"][i])
            self.array.append(dat)

    def get_clue(self):
        return [x.get_clue() for x in self.array]
    
    def get_answer(self):
        return [x.get_answer() for x in self.array]
    
    def get_type(self):
        return [x.get_type() for x in self.array]


tokenizer = T5Tokenizer.from_pretrained(T5_type)    
cl_vocab = Class_vocab(clue_type_classes)

def tokenize_for_classifier(data):
    data_class = json_data_class(data)
    output = tokenizer(data_class.get_clue(),return_tensors = 'pt',padding="max_length", truncation = True)
    output['labels'] = cl_vocab.batch_get_idx(data_class.get_type())
    return output

def tokenize_for_adapter(data):
    data_class = json_data_class(data)
    output = tokenizer(data_class.get_clue(),return_tensors = 'pt',padding="max_length", truncation = True)
    output['labels'] = tokenizer(data_class.get_answer(),return_tensors = 'pt',padding="max_length", truncation = True)['input_ids']
    output['type'] = cl_vocab.batch_get_idx(data_class.get_type())
    return output
