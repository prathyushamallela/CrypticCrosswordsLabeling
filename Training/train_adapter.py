from Models import T5modelWithAdapter
from config.configuration import T5_type, clue_type_classes,adapter_arguments
from transformers.adapters import AdapterTrainer,setup_adapter_training

T5Adapter = T5modelWithAdapter(T5_type,clue_type_classes)

setup_adapter_training(T5Adapter,adapter_arguments,clue_type_classes[0])
trainer = AdapterTrainer(model = T5Adapter)