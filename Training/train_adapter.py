from Models import T5modelWithAdapter, pre_trained_T5Tokenizer
from config.configuration import T5_type, clue_type_classes,adapter_arguments, training_arguments, max_train_samples, max_eval_samples
from transformers.adapters import AdapterTrainer,setup_adapter_training
from transformers import default_data_collator
from transformers.trainer_utils import get_last_checkpoint
import evaluate 
import os
import math

## Initialization
T5Adapter = T5modelWithAdapter(T5_type,clue_type_classes)
tokenizer = pre_trained_T5Tokenizer(T5_type)
train_dataset = None
eval_dataset = None
metric = evaluate.load("accuracy")

# Detecting last checkpoint.
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

## Preparation and initialize trainer
setup_adapter_training(T5Adapter,adapter_arguments,clue_type_classes[0])
trainer = AdapterTrainer(        
        model=T5Adapter,
        args=training_arguments,
        train_dataset=train_dataset ,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_arguments.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_arguments.do_eval else None
)

## Training
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
