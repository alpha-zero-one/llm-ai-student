import json
import os
import shutil


def message_as_html(message):
    message_fragments = message.splitlines()
    html_output = ""
    for message_fragment in message_fragments:
        html_output += f"""<p>{message_fragment}</p>"""

    return html_output


def base_to_html(path, messages):
    messages_as_html = [message_as_html(message) for message in messages]

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Markdown Display</title>
        <style>
            body {{
                font-family: Verdana, sans-serif;
                margin: 20px;
                background-color: #2b6697;
                font-size: large;
                color: white;
                text-shadow: 1px 1px black;
            }}
            .main_content {{
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .main_content_wrapper {{
                width: 100%;
                max-width: 900px;
            }}
            .blog_post {{
                background-color: rgba(255,255,255, 0.1);
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 6px;
                border-style: inset;
                border-color: steelblue;
                white-space: preserve-spaces;
            }}
            .blog_post_path {{
                font-size: medium;
                font-style: italic;
                margin-bottom: 10px;
            }}
            hr {{
                border-color: white;
            }}
            p {{
                margin-top: 0px;
                margin-bottom: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="main_content">
            <div class="main_content_wrapper">
                <h2>Evaluation Results:</h2>
                <p class="blog_post_path">
                    Path: {path}
                </p>
    """

    for message in messages_as_html:
        html_page += f"""
                    <div class="blog_post">
                        {message}
                    </div>
        """
    html_page += f"""
            </div>
        </div>
    </body>
    </html>
    """

    return html_page


if __name__ == '__main__':

    with open('training_config.json', 'r') as file:
        training_config = json.load(file)

    for base_evaluationset in training_config["base_evaluationset"]:
        for base_model in training_config["base_model"]:
            evaluation_path = f"./local/untrained/{base_model}/evaluation/{base_evaluationset}"

            with open(f"{evaluation_path}/evaluation.json", 'r') as file:
                messages_raw = json.load(file)

            messages = [message[-1]["generated_text"] for message in messages_raw]
            html_page = base_to_html(evaluation_path, messages)

            with open(f"{evaluation_path}/evaluation.html", "w") as file:
                file.write(html_page)

            summary_path = f'./local/summary/untrained/{base_model}/evaluation/{base_evaluationset}'
            if not os.path.exists(f'{summary_path}'):
                os.makedirs(f'{summary_path}')
            with open(f"{summary_path}/evaluation.json", 'w') as file:
                json.dump(messages_raw, file, indent=4)
            with open(f"{summary_path}/evaluation.html", "w") as file:
                file.write(html_page)

    for base_evaluationset in training_config["base_evaluationset"]:
        for base_dataset in training_config["base_dataset"]:
            for base_model in training_config["base_model"]:
                for lora_dropout in training_config["lora_dropout"]:
                    for lora_rank_alpha in training_config["lora_rank_alpha"]:
                        lora_rank = lora_rank_alpha["lora_rank"]
                        lora_alpha = lora_rank_alpha["lora_alpha"]
                        for learning_rate in training_config["learning_rate"]:
                            num_epochs = training_config["num_epochs"]
                            trained_path = (f"./local/trained/{base_model}/{base_dataset}/"
                                            f"lora_rank/{lora_rank}/lora_alpha/{lora_alpha}/lora_dropout/{lora_dropout}/"
                                            f"learning_rate/{learning_rate}/num_epochs/{num_epochs}")
                            evaluation_path = f"{trained_path}/evaluation/{base_evaluationset}"

                            with open(f"{evaluation_path}/evaluation.json", 'r') as file:
                                messages_raw = json.load(file)

                            messages = [message[-1]["generated_text"] for message in messages_raw]
                            html_page = base_to_html(evaluation_path, messages)

                            with open(f"{evaluation_path}/evaluation.html", "w") as file:
                                file.write(html_page)

                            summary_path = (f"./local/summary/trained/{base_model}/{base_dataset}/"
                                            f"lora_rank/{lora_rank}/lora_alpha/{lora_alpha}/lora_dropout/{lora_dropout}/"
                                            f"learning_rate/{learning_rate}/num_epochs/{num_epochs}/"
                                            f"evaluation/{base_evaluationset}")
                            if not os.path.exists(f'{summary_path}'):
                                os.makedirs(f'{summary_path}')
                            with open(f"{summary_path}/evaluation.json", 'w') as file:
                                json.dump(messages_raw, file, indent=4)
                            with open(f"{summary_path}/evaluation.html", "w") as file:
                                file.write(html_page)

    shutil.make_archive(f"./local/summary", 'zip', "./local/summary")
