from transformers import pipeline

if __name__ == '__main__':

    repo = "unsloth/Llama-3.2-1B-Instruct"
    pipe = pipeline(task="text-generation", model=f"./download/{repo}", max_length=100, truncation=True, device=1)

    response = pipe("What is a Hello World program?")
    print(response[0]["generated_text"])
