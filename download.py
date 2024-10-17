from huggingface_hub import snapshot_download

repo = "unsloth/Llama-3.2-1B-Instruct"
snapshot_download(repo_id=f"{repo}", local_dir=f"./local/download/model/{repo}")
