from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

if __name__ == '__main__':

    repo = "unsloth/Llama-3.2-1B-Instruct"
    dataset_json = "uni/student"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        f"./local/download/model/{repo}",
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="eager"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj']
    )
    peft_model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=f"./local/download/dataset/{dataset_json}/dataset.jsonl", split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    sft_config = SFTConfig(
        output_dir=f"./local/trained/training/{repo}",
        save_steps=20000000,
        max_seq_length=500,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=4,
        eval_strategy="epoch",
        learning_rate=2e-5
    )
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        args=sft_config
    )

    trainer.train()
    peft_model.save_pretrained(f"./local/trained/{repo}")

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1