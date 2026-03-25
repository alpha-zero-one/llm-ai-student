from openai import OpenAI
from pydantic import BaseModel
import difflib


class AnswerFormat(BaseModel):
    reasoning: str
    classification: int

class GPTTutor:

    def __init__(self):
        self.client = OpenAI(api_key = 'YOUR_KEY_HERE')

    def compute_diff(self, old_code, new_code):
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old_code",
            tofile="new_code",
            n=0,
            lineterm=''
        )

        result = '\n'.join(diff)
        return result

    def template(self, old_code, commit, new_code):
        diff = self.compute_diff(old_code, new_code)

        prompt = f"""You validate commits. Assume <old_code> and <new_code> are exactly handed in like this. Consider the changes in <code_difference>. Solve the tasks below step by step.

First: Find all changes made by <new_code>. Create a list of all these changes briefly. Utilize all changes mentioned in <code_difference>.

Second: The <new_code> must implement the <commit>. In this context, what are undocumented changes?

Third: Consider the list from the first task. If there are any undocumented changes mark them. Create a table with columns (change, mentioned in commit?, undocumented?).

Fourth: Return the number that fits best.
Return 0: The <commit> message has no undocumented changes AND <new_code> implemented all <commit> messages.
Return 1: There is at least one undocumented change.
Return 2: At least one implementation of a <commit> message is missing in <new_code>.
Return 3: The <commit> message does not fit the implemented changes of <new_code> at all!

<old_code>
{old_code}
</old_code>

<commit>
{commit}
</commit>

<new_code>
{new_code}
</new_code>

<code_difference>
{diff}
</code_difference>"""

        return prompt

    def evaluate(self, text):
        messages = [
            {"role": "system", "content": "You are a helpful commit validator and classifier."},
            {"role": "user", "content": text}
        ]
        chat = self.client.responses.parse(
            model="gpt-4.1-mini",
            input=messages,
            text_format=AnswerFormat,
        )

        reply = chat.output_parsed
        return reply
