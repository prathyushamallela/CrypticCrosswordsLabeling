import sys
import pathlib
sys.path.insert(0,str(pathlib.Path().cwd()))

from config.configuration import clue_type_classes, data_file_path, batch_size, dev
from torch.utils.data import  DataLoader
from utils.preprocess_data import cl_vocab, tokenize_for_adapter
from utils.save_load_model import load_solver_model
from datasets import load_dataset

class metrics():

	def __init__(self):
		self.total = 0
		self.correct_top1 = 0
		self.correct_top10 = 0
		self.correct_length = 0
		self.correct_word_count = 0

	def compute(self,model_output,answer,topk):
		self.total += len(answer)
		for i in range(len(answer)):
			output = model_output[topk*i: (i+1)*topk]
			self.top1(output,answer[i])
			self.top10(output,answer[i])
			self.length(output,answer[i])
			self.word_count(output,answer[i])
	
	def top1(self,model_output,answer):
		if answer in model_output[0]:
			self.correct_top1 += 1

	def top10(self,model_output,answer):
		found = False
		for output in model_output:
			if answer in output:
				found = True
				break
		if found:
			self.correct_top10 += 1

	def length(self,model_output,answer):
		output_length = len(model_output[0])
		answer_length = len(answer)
		if output_length == answer_length:
			self.correct_length += 1
	
	def word_count(self,model_output,answer):
		model_word_count = len(model_output[0].split())
		answer_word_count = len(answer.split())
		if model_word_count == answer_word_count:
			self.correct_word_count += 1

	def log(self):
		print("Model Evaluation :")
		print("Accuracy (Top 1) : ", (self.correct_top1/self.total)* 100,"% ")
		print("Top 10 : ", (self.correct_top10/self.total)* 100,"% ")
		print("Correct Length : ", (self.correct_length/self.total)* 100,"% ")
		print("Correct Word Counts : ", (self.correct_word_count/self.total)* 100,"% ")

# def evaluate_model(test_dataset,model):
# 	metric = metrics()
# 	for cl in clue_type_classes:
# 		dataset = test_dataset.filter(lambda example: example["type"]==cl_vocab.get_idx(cl))
# 		for i,x in enumerate(dataset):
# 			output = model(x['cluename'],topk = 10)
# 			answer = x['answer']
# 			metric.compute(output,answer)
# 	metric.log()


def evaluate_model(test_dataset,model, topk):
	metric = metrics()
	for cl in clue_type_classes:
		dataset = test_dataset.filter(lambda example: example["type"]==cl_vocab.get_idx(cl))
		dataloader = DataLoader(dataset,batch_size= batch_size, shuffle= True)
		if len(dataset):
			for i, x in enumerate(dataloader):
				output = model(x['cluename'],topk = topk)
				answer = x['answer']
				metric.compute(output,answer,topk)
	metric.log()

model = load_solver_model()
test_ds = load_dataset("json",data_files = str(data_file_path/"cryptonite-test_1679516821.180687.jsonl") , split = 'train')
test_ds = test_ds.map(tokenize_for_adapter,batched= True)
evaluate_model(test_ds, model,1)