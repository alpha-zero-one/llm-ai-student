import json
import os
import time
import subprocess
import logging
import sys

if __name__ == '__main__':

    if not os.path.exists(f'./local/log'):
        os.makedirs(f'./local/log')

    logging.basicConfig(
        filename='./local/log/training_pipeline_log.txt',
        encoding = 'utf-8',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with open('training_config.json', 'r') as file:
        training_config = json.load(file)

    progress_bar_counter = 0
    progress_bar_maximum = 1
    progress_bar_maximum *= len(training_config["base_model"])
    progress_bar_maximum *= len(training_config["code_type"])
    progress_bar_maximum *= len(training_config["solution_type"])
    progress_bar_maximum *= len(training_config["prune_type"])
    progress_bar_maximum *= len(training_config["lora_dropout"])
    progress_bar_maximum *= len(training_config["lora_rank_alpha"])
    progress_bar_maximum *= len(training_config["learning_rate"])

    dataset_name = training_config["dataset_name"]
    for base_model in training_config["base_model"]:
        for code_type in training_config["code_type"]:
            for solution_type in training_config["solution_type"]:
                for prune_type in training_config["prune_type"]:
                    for lora_dropout in training_config["lora_dropout"]:
                        for lora_rank_alpha in training_config["lora_rank_alpha"]:
                            lora_rank = lora_rank_alpha["lora_rank"]
                            lora_alpha = lora_rank_alpha["lora_alpha"]
                            for learning_rate in training_config["learning_rate"]:
                                model_name = base_model
                                model_type = "base"
                                model_path = f"./local/download/model/{model_name}"
                                max_length = training_config["max_length"]
                                num_epochs = training_config["num_epochs"]
                                trained_path = (f"./local/trained/{base_model}/{code_type}/{solution_type}/{prune_type}/"
                                                f"lora_rank/{lora_rank}/lora_alpha/{lora_alpha}/lora_dropout/{lora_dropout}/"
                                                f"learning_rate/{learning_rate}/num_epochs/{num_epochs}")
                                dataset_path = f"./local/download/dataset/{dataset_name}/{code_type}/{solution_type}/{prune_type}/dataset.jsonl"

                                progress_bar_counter += 1
                                print(f'Progress: {progress_bar_counter}/{progress_bar_maximum}')
                                print(f'Start: {time.time()}')
                                if not os.path.exists(f'{trained_path}/error'):
                                    os.makedirs(f'{trained_path}/error')

                                    try:
                                        result = subprocess.run([
                                            "python", "training_pipeline_subprocess.py",
                                             "--model_name", f"{model_name}",
                                             "--model_type", f"{model_type}",
                                             "--model_path", f"{model_path}",
                                             "--trained_path", f"{trained_path}",
                                             "--dataset_path", f"{dataset_path}",
                                             "--max_length", f"{max_length}",
                                             "--lora_rank", f"{lora_rank}",
                                             "--lora_alpha", f"{lora_alpha}",
                                             "--lora_dropout", f"{lora_dropout}",
                                             "--learning_rate", f"{learning_rate}",
                                             "--num_epochs", f"{num_epochs}"
                                            ],
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        )
                                    except subprocess.CalledProcessError as e:
                                        logging.error(f"Error running:\n{trained_path}")
                                        logging.error(f"Exit code: {e.returncode}")
                                        logging.error(f"Error output:\n{e.stderr}")
                                        sys.exit(1)

                                print(f'End: {time.time()}')
