import json
from config.configuration import data_file_path

def load_jsonl(filename):
    f = open(data_file_path/filename,'r')
    data = []
    for line in f:
        data.append(json.loads(line))
    return data