import argparse
import logging

from TrainerBase import TrainerBase
from TrainerInstruct import TrainerInstruct

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model_name', type=str,  required=True)
    argument_parser.add_argument('--model_type', type=str,  required=True)
    argument_parser.add_argument('--model_path', type=str, required=True)
    argument_parser.add_argument('--trained_path', type=str, required=True)
    argument_parser.add_argument('--dataset_path', type=str, required=True)
    argument_parser.add_argument('--learning_rate', type=float, required=True)
    argument_parser.add_argument('--num_epochs', type=int, required=True)
    args = argument_parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    model_path = args.model_path
    trained_model_path = f"{args.trained_path}/model"
    training_path = f"{args.trained_path}/training"
    dataset_path = args.dataset_path
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    logging.basicConfig(
        filename=f"{args.trained_model_path}/error/log.txt",
        encoding = 'utf-8',
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(
        f'Configuration>\n'
        f'model_name: {model_name}\n'
        f'model_type: {model_type}\n'
        f'trained_path: {trained_model_path}\n'
        f'training_path: {training_path}\n'
        f'dataset_path: {dataset_path}\n'
        f'learning_rate: {learning_rate}\n'
        f'num_epochs: {num_epochs}\n'
        f'<Configuration\n'
    )

    match model_type:
        case 'base':
            trainer = TrainerBase(
                model_name,
                model_path,
                trained_model_path,
                training_path,
                dataset_path,
                learning_rate,
                num_epochs
            )
        case 'instruct':
            trainer = TrainerInstruct(
                model_name,
                model_path,
                trained_model_path,
                training_path,
                dataset_path,
                learning_rate,
                num_epochs
            )
        case _:
            raise ValueError(f'Value "{model_type}" is illegal. Value of model_type must be "base" or "instruct".')

    try:
        trainer.train()
    except:
        logging.exception('Error output:')
        raise
