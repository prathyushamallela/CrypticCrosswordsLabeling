from transformers import T5Tokenizer
from config.configuration import T5_type, clue_type_classes
from Models.T5model import Class_vocab

class json_data_class():
    
    def __init__(self,json_data):
        self.cluename = json_data["cluename"]
        self.answer = json_data["answer"]
        self.length = json_data["length"]
        self.scores = dict()
        self.scores["anagram"] = json_data["anagram_score"]
        self.scores["container"] = json_data["container_score"]
        self.scores["reversal"] = json_data["reversal_score"]
        self.scores["deletion"] = json_data["deletion_score"]
        self.scores["homophone"] = json_data["homophone_score"]
        self.scores["charade"] = json_data["charade_score"]
        self.scores["hidden-word"] = json_data["hidden_word_score"]
        self.scores["double-def"] = json_data["double_definition_score"]
        self.scores["unclassified"] = json_data["unclassified_score"]
        self.predicted_cluetype = json_data["predicted_cluetype"]

    def get_clue(self):
        return self.cluename
    
    def get_answer(self):
        return self.answer
    
    def get_type(self):
        return max(self.scores, key = self.scores.get)


tokenizer = T5Tokenizer.from_pretrained(T5_type)    
cl_vocab = Class_vocab(clue_type_classes)

def tokenize_for_classifier(data):
    data_class = json_data_class(data)
    output = tokenizer(data_class.get_clue(),return_tensors = 'pt',padding = True, truncation = True)
    output['label'] = cl_vocab.get_idx(data_class.get_type())
    return output

def tokenize_for_adapter(data):
    data_class = json_data_class(data)
    output = tokenizer(data_class.get_clue(),return_tensors = 'pt',padding = True, truncation = True)
    output['label'] = tokenizer(data_class.get_answer(),return_tensors = 'pt',padding = True, truncation = True)
    output['type'] = cl_vocab.get_idx(data_class.get_type())
    return output



