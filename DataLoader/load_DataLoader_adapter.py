from datasets import load_dataset
from config.configuration import data_file_path, data_file_name, test_size
from utils.preprocess_data import tokenize_for_adapter, cl_vocab

train_ds = load_dataset("json",data_files = str(data_file_path/data_file_name) , split = 'train')
train_ds = train_ds.map(tokenize_for_adapter,batched= True)

anagram_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("anagram")).train_test_split(test_size=test_size)
hidden_word_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("hidden-word")).train_test_split(test_size=test_size)
container_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("container")).train_test_split(test_size=test_size)
reversal_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("reversal")).train_test_split(test_size=test_size)
deletion_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("deletion")).train_test_split(test_size=test_size)
homophone_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("homophone")).train_test_split(test_size=test_size)
double_def_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("double-def")).train_test_split(test_size=test_size)
charade_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("charade")).train_test_split(test_size=test_size)
unclassified_dataset = train_ds.filter(lambda example: example["type"]==cl_vocab.get_idx("unclassified")).train_test_split(test_size=test_size)

anagram_train_dataset = anagram_dataset['train']
anagram_eval_dataset = anagram_dataset['test']
hidden_word_train_dataset = hidden_word_dataset['train']
hidden_word_eval_dataset = hidden_word_dataset['test']
container_train_dataset = container_dataset['train']
container_eval_dataset = container_dataset['test']
reversal_train_dataset = reversal_dataset['train']
reversal_eval_dataset = reversal_dataset['test']
deletion_train_dataset = deletion_dataset['train']
deletion_eval_dataset = deletion_dataset['test']
homophone_train_dataset = homophone_dataset['train']
homophone_eval_dataset = homophone_dataset['test']
double_def_train_dataset = double_def_dataset['train']
double_def_eval_dataset = double_def_dataset['test']
charade_train_dataset = charade_dataset['train']
charade_eval_dataset = charade_dataset['test']
unclassified_train_dataset = unclassified_dataset['train']
unclassified_eval_dataset = unclassified_dataset['test']