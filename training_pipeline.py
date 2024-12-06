import json
import os
import datetime
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
    progress_bar_maximum *= len(training_config["base_dataset"])
    progress_bar_maximum *= len(training_config["base_model"])
    progress_bar_maximum *= len(training_config["lora_dropout"])
    progress_bar_maximum *= len(training_config["lora_rank_alpha"])
    progress_bar_maximum *= len(training_config["learning_rate"])

    base_dataset = training_config["base_dataset"]
    for base_model in training_config["base_model"]:
        for lora_dropout in training_config["lora_dropout"]:
            for lora_rank_alpha in training_config["lora_rank_alpha"]:
                lora_rank = lora_rank_alpha["lora_rank"]
                lora_alpha = lora_rank_alpha["lora_alpha"]
                for learning_rate in training_config["learning_rate"]:
                    model_type = "base"
                    dataset_path = f"./local/download/dataset/{base_dataset}/dataset.jsonl"
                    model_path = f"./local/download/model/{base_model}"
                    max_length = training_config["max_length"]
                    num_epochs = training_config["num_epochs"]
                    trained_path = (f"./local/trained/{base_model}/{base_dataset}/"
                                    f"lora_rank/{lora_rank}/lora_alpha/{lora_alpha}/lora_dropout/{lora_dropout}/"
                                    f"learning_rate/{learning_rate}/num_epochs/{num_epochs}")

                    progress_bar_counter += 1
                    print(f'Progress: {progress_bar_counter}/{progress_bar_maximum}')
                    datetime_now = datetime.datetime.now()
                    time = datetime_now.strftime("%Y-%m-%d %H:%M:%S")
                    print(f'Start: {time}')
                    if not os.path.exists(f'{trained_path}/error'):
                        os.makedirs(f'{trained_path}/error')

                        try:
                            result = subprocess.run([
                                "python", "training_pipeline_subprocess.py",
                                 "--model_name", f"{base_model}",
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

                    datetime_now = datetime.datetime.now()
                    time = datetime_now.strftime("%Y-%m-%d %H:%M:%S")
                    print(f'End: {time}')
