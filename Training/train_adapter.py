from Models import T5modelWithAdapter, pre_trained_T5Tokenizer
from config.configuration import T5_type, clue_type_classes,adapter_arguments, training_arguments
from transformers.adapters import AdapterTrainer,setup_adapter_training


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
        # data_collator=default_data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        # if training_args.do_eval and not is_torch_tpu_available()
        # else None
)