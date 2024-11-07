
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format
import bitsandbytes as bnb


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class TrainerInstruct:

    def __init__(
        self,
        model_name,
        model_path,
        trained_model_path,
        training_path,
        dataset_path,
        learning_rate,
        num_epochs
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.trained_model_path = trained_model_path
        self.training_path = training_path
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(self):
        #TODO implement
        pass

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
