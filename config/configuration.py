import pathlib as Path
from transformers.adapters import PrefixTuningConfig 
from transformers.adapters.training import AdapterArguments

clue_type_classes = ["anagram","hidden-word","container","reversal","deletion","homophone","double-def","charade","unclassified"]

T5_type = 't5-small'
All_T5_types = ['t5-small','t5-base','t5-large','t5-3b','t5-11b']

mock_input = ["translate english to mandarin: This is my data","This is a mock data, I am testing.","Another mock data sentence, also for testing","Another one for testing"]

save_file_path = Path.Path().cwd() / 'Model_saveflies'
adapter_config = PrefixTuningConfig(flat = False, prefix_length = 30)
adapter_arguments = AdapterArguments(train_adapter = True,adapter_config =adapter_config)