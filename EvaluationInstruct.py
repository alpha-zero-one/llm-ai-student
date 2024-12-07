import torch
import json
import os
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


class EvaluationInstruct:

    def __init__(
            self,
            model_name,
            model_path,
            use_adapter,
            trained_model_path,
            evaluation_path,
            evaluationset_path,
            max_length
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.use_adapter = use_adapter
        self.trained_model_path = trained_model_path
        self.evaluation_path = evaluation_path
        self.evaluationset_path = evaluationset_path
        self.max_length = max_length

    def evaluate(self):
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

        if self.use_adapter:
            model.load_adapter(self.trained_model_path)

        '''
        generation_config = model.generation_config
        generation_config.max_new_tokens=300
        generation_config.do_sample=True
        generation_config.temperature=1.0
        '''

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )

        with open(self.evaluationset_path, 'r') as file:
            input_text = json.load(file)

        response = pipe(input_text, max_new_tokens=300)

        os.makedirs(self.evaluation_path, exist_ok=True)
        with open(f"{self.evaluation_path}/evaluation.json", 'w') as file:
            json.dump(response, file, indent=4)
