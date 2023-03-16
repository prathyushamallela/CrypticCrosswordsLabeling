from Models import T5modelWithAdapter
from config.configuration import T5_type, clue_type_classes,adapter_arguments

T5Adapter = T5modelWithAdapter(T5_type,clue_type_classes)
