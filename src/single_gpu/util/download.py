import json
from huggingface_hub import snapshot_download

with open("./config/training_config.json", 'r') as file:
    training_config = json.load(file)

    for repo in training_config["base_model"]:
        snapshot_download(repo_id=f"{repo}", local_dir=f"./local/download/model/{repo}")

    for repo in training_config["instruct_model"]:
        snapshot_download(repo_id=f"{repo}", local_dir=f"./local/download/model/{repo}")
