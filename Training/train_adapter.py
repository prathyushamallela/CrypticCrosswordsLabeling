import sys
import pathlib
sys.path.insert(0,str(pathlib.Path().cwd()))

from Models.T5model import T5modelWithAdapter, pre_trained_T5Tokenizer
from config.configuration import T5_type, clue_type_classes,adapter_arguments, training_arguments, max_train_samples, max_eval_samples
from transformers.adapters import AdapterTrainer,setup_adapter_training
from transformers import default_data_collator
from transformers.trainer_utils import get_last_checkpoint
import evaluate 
import os
import math
from DataLoader.load_DataLoader_adapter import anagram_train_dataset, anagram_eval_dataset,hidden_word_train_dataset, hidden_word_eval_dataset, container_train_dataset, container_eval_dataset,reversal_train_dataset,reversal_eval_dataset,deletion_train_dataset,deletion_eval_dataset ,homophone_train_dataset,homophone_eval_dataset,double_def_train_dataset,double_def_eval_dataset,charade_train_dataset,charade_eval_dataset,unclassified_train_dataset,unclassified_eval_dataset

## Initialization
T5Adapter = T5modelWithAdapter(T5_type,clue_type_classes)
tokenizer = pre_trained_T5Tokenizer(T5_type).T5tokenizer
metric = evaluate.load("accuracy")

# Detecting last checkpoint.
def get_checkpoint():
        last_checkpoint = None
        if os.path.isdir(training_arguments.output_dir) and training_arguments.do_train and not training_arguments.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(training_arguments.output_dir)
                if last_checkpoint is None and len(os.listdir(training_arguments.output_dir)) > 0:
                        raise ValueError(
                        f"Output directory ({training_arguments.output_dir}) already exists and is not empty. "
                        "Use --overwrite_output_dir to overcome."
                        )
                elif last_checkpoint is not None and training_arguments.resume_from_checkpoint is None:
                        print("Checkpoint detected. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        return last_checkpoint

def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
        return logits.argmax(dim=-1)

def prepare_trainer(model,train_dataset,eval_dataset,tokenizer,task_name):
## Preparation and initialize trainer
        setup_adapter_training(model,adapter_arguments,task_name)
        trainer = AdapterTrainer(        
                model=model,
                args=training_arguments,
                train_dataset=train_dataset ,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=default_data_collator,
                compute_metrics=compute_metrics if training_arguments.do_eval else None,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_arguments.do_eval else None
        )
        return trainer

def train(trainer,train_dataset):
        ## Training    
        last_checkpoint = get_checkpoint()
        checkpoint = None
        if training_arguments.do_train:
                checkpoint = None
                if training_arguments.resume_from_checkpoint is not None:
                        checkpoint = training_arguments.resume_from_checkpoint
                elif last_checkpoint is not None:
                        checkpoint = last_checkpoint
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
                trainer.save_model()  # Saves the tokenizer too for easy upload

                metrics = train_result.metrics

                metrics["train_samples"] = min(max_train_samples, len(train_dataset))

                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()
        return trainer

def eval(trainer,eval_dataset):
        # Evaluation
        if training_arguments.do_eval:
                print("*** Evaluate ***")

                metrics = trainer.evaluate()

                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                try:
                        perplexity = math.exp(metrics["eval_loss"])
                except OverflowError:
                        perplexity = float("inf")
                metrics["perplexity"] = perplexity

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
        return trainer

## Anagram
if len(anagram_train_dataset)+ len(anagram_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,anagram_train_dataset,anagram_eval_dataset,tokenizer,"anagram")
        trainer = train(trainer,anagram_train_dataset)
        trainer = eval(trainer,anagram_eval_dataset)

## Hidden-word
if len(hidden_word_train_dataset)+ len(hidden_word_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,hidden_word_train_dataset,hidden_word_eval_dataset,tokenizer,"hidden-word")
        trainer = train(trainer,hidden_word_train_dataset)
        trainer = eval(trainer,hidden_word_eval_dataset)

## container
if len(container_train_dataset)+ len(container_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,container_train_dataset,container_eval_dataset,tokenizer,"container")
        trainer = train(trainer,container_train_dataset)
        trainer = eval(trainer,container_eval_dataset)

## reversal
if len(reversal_train_dataset)+ len(reversal_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,reversal_train_dataset,reversal_eval_dataset,tokenizer,"reversal")
        trainer = train(trainer,reversal_train_dataset)
        trainer = eval(trainer,reversal_eval_dataset)

## deletion
if len(deletion_train_dataset)+ len(deletion_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,deletion_train_dataset,deletion_eval_dataset,tokenizer,"deletion")
        trainer = train(trainer,deletion_train_dataset)
        trainer = eval(trainer,deletion_eval_dataset)

## homophone
if len(homophone_train_dataset)+ len(homophone_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,homophone_train_dataset,homophone_eval_dataset,tokenizer,"homophone")
        trainer = train(trainer,homophone_train_dataset)
        trainer = eval(trainer,homophone_eval_dataset)

## double_definition
if len(double_def_train_dataset)+ len(double_def_eval_dataset)>0:     
        trainer = prepare_trainer(T5Adapter.model,double_def_train_dataset,double_def_eval_dataset,tokenizer,"double_definition")
        trainer = train(trainer,double_def_train_dataset)
        trainer = eval(trainer,double_def_eval_dataset)

## charade
if len(charade_train_dataset)+ len(charade_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,charade_train_dataset,charade_eval_dataset,tokenizer,"charade")
        trainer = train(trainer,charade_train_dataset)
        trainer = eval(trainer,charade_eval_dataset)

# unclassified
if len(unclassified_train_dataset)+ len(unclassified_eval_dataset)>0:
        trainer = prepare_trainer(T5Adapter.model,unclassified_train_dataset,unclassified_eval_dataset,tokenizer,"unclassified")
        trainer = train(trainer,unclassified_train_dataset)
        trainer = eval(trainer,unclassified_eval_dataset)