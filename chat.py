import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':

    repo = "unsloth/Llama-3.2-1B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        f"./download/{repo}",
        quantization_config=quantization_config
    )
    model.load_adapter(f"./trained/{repo}")
    tokenizer = AutoTokenizer.from_pretrained(f"./download/{repo}")
    pipe = pipeline(
        task="text-generation", max_new_tokens=1000,
        model=model, tokenizer=tokenizer,
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

            response = pipe(chat)
            chat = response[-1]["generated_text"]
            answer = chat[-1]["content"]

            print(f"AI: {answer}")
            print()
