
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
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


class TrainerBase:

    def __init__(
        self,
        model_name,
        model_path,
        trained_model_path,
        training_path,
        dataset_path,
        max_length,
        lora_rank,
        lora_alpha,
        lora_dropout,
        learning_rate,
        num_epochs
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.trained_model_path = trained_model_path
        self.training_path = training_path
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )

        modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules
        )
        peft_model = get_peft_model(model, lora_config)

        dataset = load_dataset(
            "json",
            data_files=self.dataset_path,
            split="train"
        )
        #dataset = dataset.select(range(1000))
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)

        sft_config = SFTConfig(
            output_dir=self.training_path,
            save_steps=1000,
            #eval_strategy="steps",
            #eval_steps=50,
            dataset_text_field="text",
            max_seq_length=self.max_length,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            packing=False,
            group_by_length=False,
            optim="adamw_8bit",
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            fp16=False,
            bf16=False
        )

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
            args=sft_config
        )

        trainer.train()
        peft_model.save_pretrained(self.trained_model_path)

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
