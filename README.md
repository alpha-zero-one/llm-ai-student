# llm-ai-student

Build, finetune, run a LLM to simulate a profile of an AI student.

## Installation

Install a version of pytorch for your system from:<br>
https://pytorch.org/get-started/locally/

Install the requirements. Using pip:

```bash
pip install -r "requirements.txt"
```

## Folder Structure

The application follows the following folder structure:

- config/
  - fsdp_config.yaml
  - training_config.json
- local/
  - download/
    - dataset/
      - dataset/path/here/
        - dataset.jsonl
    - log/
      - evaluation_pipeline_log.txt
      - training_pipeline_log.txt
    - evaluationset/
      - test/
        - evalulationset/path/here/
            - evaluationset.json
    - model/
      - user/model/
        - model_files_here
  - trained/
    - user/model/
      - dataset_path/
        - lora_rank/value/
        - lora_alpha/value/
        - lora_dropout/value/
        - learning_rate/value/
        - num_epochs/value/
          - error/
            - evaluation_log.txt
            - log.txt
          - evaluation/evaluationset_path/
            - evaluation.json
          - model/
            - model_files_here
          - training/
            - checkpoints/here/
  - untrained/
    - user/model/
      - error/evaluation_log.txt
      - evaluation/evaluationset_path/
        - evaluation.json

## Application

Edit the `training_config.json` to select the LLM, dataset and evaluationset, and hyperparameters.
Edit the `fsdp_config.yaml` to select the number of gpu `num_processes`.

Step 1: Download the `base_model` and `instruct_model` with:

```bash
python3 "./src/single_gpu/util/download.py"
```

Step 2: Upload your dataset and evaluationset as described in the folder structure. The `dataset.jsonl` contains samples of the form:

```jsonl
{"text": "This is the text that the model is supposed to learn."}
{"text": "Here is more text that the model is supposed to learn."}
```

The `evaluationset.json` contains samples of the form:
```json
[
  "This is the text that the model",
  "Here is more text that the model is"
]
```

Step 3: Run the training*:

```bash
python3 "./src/single_gpu/training/training_pipeline.py"
```
*Note: Training will be skipped for a given configuration when under `trained/` a given `error/log.txt` exist. Empty the trained folder before restarting training.

Step 4: Run the evaluation:

```bash
python3 "./src/single_gpu/evaluate/evaluation_pipeline.py"
```

Step 5: Collect the evaluation as `summary.zip` for download:

```bash
python3 "./src/single_gpu/visualize/visualize.py"
```