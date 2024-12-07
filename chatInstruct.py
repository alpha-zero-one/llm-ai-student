import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':

    use_adapter = False

    #model_path = "./local/download/model/unsloth/Llama-3.2-3B"
    #adapter_path = "./local/trained/unsloth/Llama-3.2-3B/model"
    model_path = "./local/download/model/Qwen/Qwen2.5-Coder-7B"
    adapter_path = "./local/trained/Qwen/Qwen2.5-Coder-7B/model"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )

    if use_adapter:
        model.load_adapter(adapter_path)

    '''
    generation_config = model.generation_config
    generation_config.max_new_tokens=300
    generation_config.do_sample=True
    generation_config.temperature=1.0
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    chat = [
        {
            "role": "system",
            "content": "Du bist ein Student und studierst Informatik."
        }
    ]

    running = True
    while running:
        message = input("> ")

        if message == "/bye":
            running = False
        else:
            chat.append({
                "role": "user",
                "content": f"{message}"
            })

            response = pipe(chat, max_new_tokens=300)
            chat = response[-1]["generated_text"]
            answer = chat[-1]["content"]

            print(f"AI: {answer}")
            print()
