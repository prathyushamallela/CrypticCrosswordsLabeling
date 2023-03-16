from Models import T5modelWithAdapter, pre_trained_T5Tokenizer
from config.configuration import T5_type, clue_type_classes,adapter_arguments, training_arguments
from transformers.adapters import AdapterTrainer,setup_adapter_training

def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

## Initialization
T5Adapter = T5modelWithAdapter(T5_type,clue_type_classes)
tokenizer = pre_trained_T5Tokenizer(T5_type)
train_dataset = None
eval_dataset = None

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
        compute_metrics=compute_metrics if training_arguments.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_arguments.do_eval and not is_torch_tpu_available()
        else None
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

        max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

# Evaluation
if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
                perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
                perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
                kwargs["dataset"] = data_args.dataset_name