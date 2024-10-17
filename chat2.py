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
        f"./local/download/model/{repo}",
        quantization_config=quantization_config
    )
    model.load_adapter(f"./local/trained/{repo}")
    tokenizer = AutoTokenizer.from_pretrained(f"./local/download/model/{repo}")
    pipe = pipeline(
        task="text-generation", max_new_tokens=1000,
        model=model, tokenizer=tokenizer,
        device_map="auto"
    )

    running = True
    aufgabe = input("Aufgabe > ")
    code = input("Code > ")
    while running:
        bemerkung = input("> ")

        if bemerkung == "/bye":
            running = False
        else:
            message = f'Aufgabe:\n{aufgabe}\n\nCode:\n{code}\n\nBemerkung:{bemerkung}\n\nVerbesserung:\n'
            chat = [
                {
                    "role": "system",
                    "content": "Du bist ein neuer Student und studierst Informatik."
                },
                {
                    "role": "user",
                    "content": message
                }
            ]

            response = pipe(chat)
            chat = response[-1]["generated_text"]
            code = chat[-1]["content"]

            print(f"AI: {code}")
            print()
