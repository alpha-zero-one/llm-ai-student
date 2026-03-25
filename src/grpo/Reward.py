import concurrent.futures

import logging

import json

from GPTTutor import GPTTutor


logger = logging.getLogger(__name__)
logging.basicConfig(filename='reward_error_log.txt', encoding='utf-8', level=logging.INFO)

# Note:
# - prompts is a list of conversations
# - a conversation is a list of messages
# - a message is a dict of keys (role, content)
# - completions is a list of conversations
# - prompts[i][1]["content"] contains the initial question of conversation i
# - completions[i][0]["content"] contains the response of conversation i


# Reward the form  <reasoning> ... </reasoning> <answer> ... </answer>
def validate_reasoning(text):
    try:
        reasoning = text.strip()
        if not reasoning.startswith("<reasoning>"):
            return False
        
        reasoning = reasoning.split("<reasoning>")[1]
        reasoning = reasoning.split("</reasoning>")[0]
        reasoning = reasoning.strip()

        if reasoning:
            return True
        else:
            return False
    except IndexError:
        return False
    
def reasoning_reward_func(completions:list, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a reasoning."""    
    contents = [completion[0]["content"] for completion in completions]
    reasonings = [validate_reasoning(c) for c in contents]
    rewards = [0.5 if reasoning else 0.0 for reasoning in reasonings]

    return rewards

def validate_answer(text):
    try:
        answer = text.strip()
        if not answer.endswith("</answer>"):
            return False

        answer = answer.split("<answer>")[1]
        answer = answer.split("</answer>")[0]
        answer = answer.strip()

        if answer:
            return True
        else:
            return False
    except IndexError:
        return False
    
def answer_reward_func(completions:list, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a answer."""    
    contents = [completion[0]["content"] for completion in completions]
    answers = [validate_answer(c) for c in contents]
    rewards = [0.5 if answer else 0.0 for answer in answers]

    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(completions:list, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def extract_answer(text):
    try:
        answer = text.split("<answer>")[1].split("</answer>")[0]
        answer = answer.strip()
        return answer if answer else "NO ANSWER PROVIDED."
    except IndexError:
        return "NO ANSWER PROVIDED."
    
def extract_reasoning(text):
    try:
        answer = text.split("<reasoning>")[1].split("</reasoning>")[0]
        answer = answer.strip()
        return answer if answer else "NO REASONING PROVIDED."
    except IndexError:
        return "NO REASONING PROVIDED."

# Reward length old_code similar to length new_code
def code_length_reward_func(prompts:list, completions:list, old_code:list, feedback:list, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    answers = [extract_answer(response) for response in responses]

    rewards = []
    for old_code_sample, new_code_sample in zip(old_code, answers):
        len_old = len(old_code_sample)
        len_new = len(new_code_sample)

        if len_old == 0 or len_new == 0:
            return 0.0

        percentage_difference = abs(len_old - len_new) / max(len_old, len_new)
        reward = 1.0 - (percentage_difference ** 0.9)
        reward = max(0.0, reward)

        rewards.append(reward)

    return rewards

# Reward correctness
def gpt_tutor_reward_func(prompts:list, completions:list, old_code:list, feedback:list, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    answers = [extract_answer(response) for response in responses]

    step = {
        "task": {
            "old_code": old_code[0],
            "feedback": feedback[0]
        },
        "student": {
            "full": completions[0][0]["content"],
            "reasoning": extract_reasoning(completions[0][0]["content"]),
            "answer": extract_answer(completions[0][0]["content"])
        }
    }

    gpt_tutor = GPTTutor()

    def evaluate_sample(old_code_sample, feedback_sample, new_code_sample):
        try:
            template = gpt_tutor.template(old_code_sample, feedback_sample, new_code_sample)
            evaluation = gpt_tutor.evaluate(template)
            reward = 0.0
            if evaluation.classification == 0:
                reward = 4.0
            return reward, evaluation
        except Exception:
            logging.exception("An exception occurred!")
            return 0.0, "NO EVALUATION DUE TO ERROR!"


    rewards = [0.0] * len(old_code)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_sample, o, f, a): i
            for i, (o, f, a) in enumerate(zip(old_code, feedback, answers))
        }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            reward, evaluation = future.result()
            rewards[i] = reward
            
            if i==0:
                step["tutor"] = {
                    "reasoning": evaluation.reasoning,
                    "classification": evaluation.classification
                }

    logging.info(rewards)
    print(rewards)

    with open('step.json', 'w') as json_file:
        json.dump(step, json_file)

    return rewards



def total_reward_func(prompts:list, completions:list, old_code:list, feedback:list, **kwargs):

    rewards_reasoning = reasoning_reward_func(completions)
    rewards_answer = answer_reward_func(completions)
    rewards_xmlcount = xmlcount_reward_func(completions)
    rewards_code_length = code_length_reward_func(prompts, completions, old_code, feedback)
    rewards_gpt_tutor = gpt_tutor_reward_func(prompts, completions, old_code, feedback)

    rewards = [rewards_reasoning[i] + rewards_answer[i] + rewards_xmlcount[i] + rewards_code_length[i] + rewards_gpt_tutor[i] for i in range(len(rewards_reasoning))]

    with open('rewards_collection.txt', 'a') as file:
        file.write("\n")
        file.write("\n" + json.dumps(rewards_reasoning))
        file.write("\n" + json.dumps(rewards_answer))
        file.write("\n" + json.dumps(rewards_xmlcount))
        file.write("\n" + json.dumps(rewards_code_length))
        file.write("\n" + json.dumps(rewards_gpt_tutor))
        file.write("\n" + json.dumps(rewards))

    return rewards
