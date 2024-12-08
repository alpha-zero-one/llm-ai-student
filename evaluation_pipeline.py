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
        filename='./local/log/evaluation_pipeline_log.txt',
        encoding = 'utf-8',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with open('training_config.json', 'r') as file:
        training_config = json.load(file)

    progress_bar_counter = 0
    progress_bar_maximum = 1
    progress_bar_maximum *= len(training_config["base_evaluationset"])
    progress_bar_maximum *= len(training_config["base_model"])

    for base_evaluationset in training_config["base_evaluationset"]:
        for base_model in training_config["base_model"]:
            model_type = "base"
            model_path = f"./local/download/model/{base_model}"
            max_length = training_config["max_length"]
            use_adapter = False
            trained_path = ""
            evaluation_path = f"./local/untrained/{base_model}/evaluation/{base_evaluationset}"
            evaluationset_path = f"./local/download/evaluationset/{base_evaluationset}"

            progress_bar_counter += 1
            print(f'Progress: {progress_bar_counter}/{progress_bar_maximum}')
            datetime_now = datetime.datetime.now()
            time = datetime_now.strftime("%Y-%m-%d %H:%M:%S")
            print(f'Start: {time}')

            try:
                result = subprocess.run([
                    "python", "evaluation_pipeline_subprocess.py",
                    "--model_name", f"{base_model}",
                    "--model_type", f"{model_type}",
                    "--model_path", f"{model_path}",
                    "--trained_path", f"{trained_path}",
                    "--evaluation_path", f"{evaluation_path}",
                    "--evaluationset_path", f"{evaluationset_path}",
                    "--max_length", f"{max_length}"
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


    progress_bar_counter = 0
    progress_bar_maximum = 1
    progress_bar_maximum *= len(training_config["base_evaluationset"])
    progress_bar_maximum *= len(training_config["base_dataset"])
    progress_bar_maximum *= len(training_config["base_model"])
    progress_bar_maximum *= len(training_config["lora_dropout"])
    progress_bar_maximum *= len(training_config["lora_rank_alpha"])
    progress_bar_maximum *= len(training_config["learning_rate"])

    for base_evaluationset in training_config["base_evaluationset"]:
        for base_dataset in training_config["base_dataset"]:
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
                            use_adapter = True
                            trained_path = (f"./local/trained/{base_model}/{base_dataset}/"
                                            f"lora_rank/{lora_rank}/lora_alpha/{lora_alpha}/lora_dropout/{lora_dropout}/"
                                            f"learning_rate/{learning_rate}/num_epochs/{num_epochs}")
                            evaluation_path = f"{trained_path}/evaluation/{base_evaluationset}"
                            evaluationset_path = f"./local/download/evaluationset/{base_evaluationset}/evaluationset.json"

                            progress_bar_counter += 1
                            print(f'Progress: {progress_bar_counter}/{progress_bar_maximum}')
                            datetime_now = datetime.datetime.now()
                            time = datetime_now.strftime("%Y-%m-%d %H:%M:%S")
                            print(f'Start: {time}')

                            try:
                                result = subprocess.run([
                                    "python", "evaluation_pipeline_subprocess.py",
                                        "--model_name", f"{base_model}",
                                        "--model_type", f"{model_type}",
                                        "--model_path", f"{model_path}",
                                        "--use_adapter",
                                        "--trained_path", f"{trained_path}",
                                        "--evaluation_path", f"{evaluation_path}",
                                        "--evaluationset_path", f"{evaluationset_path}",
                                        "--max_length", f"{max_length}"
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
