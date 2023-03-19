from utils.load_dataset import load_jsonl
from torch.utils.data import Dataset, DataLoader

class json_data_class():

    def __init__(self,json_data):
        self.cluename = json_data["cluenam"]
        self.answer = json_data["answer"]
        self.length = json_data["length"]
        self.anagram_score = json_data["anagram_score"]
        self.container_score = json_data["container_score"]
        self.reversal_score = json_data["reversal_score"]
        self.deletion_score = json_data["deletion_score"]
        self.homophone_score = json_data["homophone_score"]
        self.charade_score = json_data["charade_score"]
        self.hidden_word_score = json_data["hidden_word_score"]
        self.double_definition_score = json_data["double_definition_score"]
        self.unclassified_score = json_data["unclassified_score"]
        self.predicted_cluetype = json_data["predicted_cluetype"]


class crossword_dataset(Dataset):

    def __init__(self,json_data):
        self.data = [json_data_class(x) for x in json_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
train_dataloader = DataLoader(crossword_dataset(load_jsonl("sample1679170166.779008.jsonl")),batch_size= 50, shuffle= True)