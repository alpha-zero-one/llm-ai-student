import json

from unsloth import FastLanguageModel

dataset_path = './trainer/evaluationset/uni_no_reasoning/student3/evaluationset.json'
max_seq_length = 1000 # Choose any! We auto support RoPE Scaling internally!
max_new_tokens = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

with open(dataset_path, 'r') as file:
    dataset = json.load(file)

evaluation = ["" for _ in range(len(dataset))]
for index, sample in enumerate(dataset):
    messages = [
        {
            "role": "system",
            "content": "\nYou are a helpful AI and implement the task exactly as the user states.\n"
        },
        {
            "role": "user",
            "content": sample
        }
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
    outputs = model.generate(input_ids = inputs, max_new_tokens = max_new_tokens, use_cache = True)
    original_text = tokenizer.batch_decode(outputs)

    text = original_text[0]
    text = text.split("<|im_start|>assistant\n")[1]
    text = text.split("<|im_end|>")[0]
    text = text.strip()

    evaluation[index] = text

with open('evaluation.json', 'w') as json_file:
    json.dump(evaluation, json_file)
