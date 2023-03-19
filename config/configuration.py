import pathlib as Path
from transformers import TrainingArguments
from transformers.adapters import PrefixTuningConfig, AdapterArguments 

clue_type_classes = ["anagram","hidden-word","container","reversal","deletion","homophone","double-def","charade","unclassified"]

T5_type = 't5-small'
All_T5_types = ['t5-small','t5-base','t5-large','t5-3b','t5-11b']

mock_input = ["translate en to de: This is my data","This is a mock data, I am testing.","Another mock data sentence, also for testing","Another one for testing"]

save_file_path = Path.Path().cwd() / 'Model_saveflies'
data_file_path = Path.Path().cwd() / 'data'

adapter_config = PrefixTuningConfig(flat = False, prefix_length = 30)
## Refer to https://docs.adapterhub.ml/classes/adapter_training.html#transformers.adapters.training.AdapterArguments
adapter_arguments = AdapterArguments(train_adapter = True,adapter_config =adapter_config)
## Refer to https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
training_arguments = TrainingArguments(output_dir= save_file_path,do_train = True, do_eval = True)

max_train_samples = 500
max_eval_samples = 500