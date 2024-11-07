import json
import os
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

    dataset_name = training_config["dataset_name"]
    for base_model in training_config["base_model"]:
        for code_type in training_config["code_type"]:
            for solution_type in training_config["solution_type"]:
                for prune_type in training_config["prune_type"]:
                    for learning_rate in training_config["learning_rate"]:
                        model_name = base_model
                        model_type = "base"
                        model_path = f"./local/download/model/{model_name}"
                        trained_path = (f"./local/trained/{base_model}/{code_type}/{solution_type}/{prune_type}/"
                                              f"learning_rate/{learning_rate}") # continue here learning rate map?
                        dataset_path = f"./local/download/dataset/{dataset_name}/{code_type}/{solution_type}/{prune_type}/dataset.jsonl"
                        num_epochs = training_config["num_epochs"]

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
