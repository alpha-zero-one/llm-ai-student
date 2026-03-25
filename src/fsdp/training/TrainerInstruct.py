from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from accelerate import Accelerator
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


class TrainerInstruct:

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

        Accelerator()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        )

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

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
            report_to="none",
            save_steps=20000,
            #eval_strategy="steps",
            #eval_steps=50,
            max_seq_length=self.max_length,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            packing=False,
            group_by_length=False,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            fp16=False,
            bf16=True
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
            args=sft_config
        )

        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

        trainer.train()

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(self.trained_model_path)

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
