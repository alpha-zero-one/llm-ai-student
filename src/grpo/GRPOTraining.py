import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
import time
import json

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

from Reward import total_reward_func


start = time.time()

max_prompt_length = 700
max_completion_length = 2100
max_seq_length = max_prompt_length + max_completion_length
generations = 6
lora_rank = 8 # Larger rank = smarter, but slower
model_name = "Qwen/Qwen2.5-7B-Instruct"
dataset_path = './trainer2/dataset/uni/student3/dataset.jsonl'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = False, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

save_steps = 5 #save every 10 samples

#dataset_skip_list = []
#dataset_range = [i for i in range(3000) if i not in dataset_skip_list]

dataset = load_dataset(
    "json",
    data_files=dataset_path,
    split="train"
)
dataset = dataset.shuffle(seed=42).select(range(5000))
#dataset = dataset.select(dataset_range)
print(dataset)


training_args = GRPOConfig(
    use_vllm = False, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    per_device_train_batch_size = generations,
    num_generations = generations, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    save_strategy="steps",
    save_steps = save_steps,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        total_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)

if len(os.listdir('./outputs')) == 0:
    print("Folder 'outputs' is empty. Training from scratch.")
    trainer.train()
else:
    print("Folder 'outputs' has checkpoints. Training from last checkpoint.")
    trainer.train(resume_from_checkpoint=True)

model.save_pretrained("model")
tokenizer.save_pretrained("model")

end = time.time()
duration = end - start
print(duration)

with open('duration.json', 'w') as json_file:
    json.dump(duration, json_file)
    