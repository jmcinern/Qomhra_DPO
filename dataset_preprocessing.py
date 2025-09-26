from datasets import load_dataset
import pandas as pd
import re
from transformers import AutoTokenizer

with open("secrets.json", "r") as f:
    token = eval(f.read())["HF_KEY"]
# columns should be "chosen" and "rejected" for trainer 
# data should have chat template applied

REPO_ID = "jmcinern/LIMA_ga" 
FILE_NAME = "translated_IRT_ga.jsonl" 

dataset = load_dataset(
    "json", 
    data_files=f"hf://datasets/{REPO_ID}/{FILE_NAME}",
    token=token
)

# convert to DPO trainer format
instructions = []
chosens = []
rejecteds = []

# prepare data to apply chat template
tokenizer = AutoTokenizer.from_pretrained(
"Qwen/Qwen3-8B", 
trust_remote_code=True, 
)

def custom_qwen_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
    formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=tokenize, 
                    add_generation_prompt=add_generation_prompt, 
                    enable_thinking=enable_thinking
                )
    # Remove  <think>...</think> tags (enable_thinking=False doesnt seem to be effective)
    think_regex = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
    formatted_text = think_regex.sub("", formatted_text)
    prompt, response = formatted_text.split("<|im_start|>assistant\n")
    return prompt, response

'''
dpo_dataset_dict = {
    "prompt": [],
    "chosen": [],
    "rejected": []
'''
prompts = []
chosens = []
rejecteds = []  
for item in dataset["train"]:
        ''' Take the instruction and response and 
            formats them with custom non-thinking Qwen-3 chat template.
            instruction, response1, response2
        '''
        messages_chosen = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response1"]}
        ]

        messages_rejected = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response2"]}
        ]

        prompt, chosen_formatted = custom_qwen_chat_template(messages_chosen)
        rejected_formatted = custom_qwen_chat_template(messages_rejected)

        prompts.append(prompt)
        chosens.append(chosen_formatted)
        rejecteds.append(rejected_formatted)

dpo_dataset = pd.DataFrame({
    "prompt": prompts,
    "chosen": chosens,
    "rejected": rejecteds
})

with open("dpo_dataset_ga.csv", "w", encoding="utf-8") as f:
    dpo_dataset.to_csv(f, index=False)