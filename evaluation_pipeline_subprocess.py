import argparse
import logging

from EvaluationBase import EvaluationBase
from EvaluationInstruct import EvaluationInstruct

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model_name', type=str,  required=True)
    argument_parser.add_argument('--model_type', type=str,  required=True)
    argument_parser.add_argument('--model_path', type=str, required=True)
    argument_parser.add_argument('--use_adapter', action="store_true")
    argument_parser.add_argument('--trained_path', type=str, required=True)
    argument_parser.add_argument('--evaluation_path', type=str, required=True)
    argument_parser.add_argument('--evaluationset_path', type=str, required=True)
    argument_parser.add_argument('--max_length', type=int, required=True)
    args = argument_parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    model_path = args.model_path
    use_adapter = args.use_adapter
    trained_model_path = f"{args.trained_path}/model"
    evaluation_path = args.evaluation_path
    evaluationset_path = args.evaluationset_path
    max_length = args.max_length

    if use_adapter:
        logging.basicConfig(
            filename=f"{args.trained_path}/error/evaluation_log.txt",
            encoding = 'utf-8',
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        logging.info(
            f'Configuration>\n'
            f'model_name: {model_name}\n'
            f'model_type: {model_type}\n'
            f'model_path: {model_path}\n'
            f'model_path: {use_adapter}\n'
            f'trained_model_path: {trained_model_path}\n'
            f'evaluation_path: {evaluation_path}\n'
            f'evaluationset_path: {evaluationset_path}\n'
            f'max_length: {max_length}\n'
            f'<Configuration\n'
        )

    match model_type:
        case 'base':
            evaluate = EvaluationBase(
                model_name,
                model_path,
                use_adapter,
                trained_model_path,
                evaluation_path,
                evaluationset_path,
                max_length
            )
        case 'instruct':
            evaluate = EvaluationInstruct(
                model_name,
                model_path,
                use_adapter,
                trained_model_path,
                evaluation_path,
                evaluationset_path,
                max_length
            )
        case _:
            raise ValueError(f'Value "{model_type}" is illegal. Value of model_type must be "base" or "instruct".')

    try:
        evaluate.evaluate()
    except:
        if use_adapter:
            logging.exception('Error output:')
            raise
