from datasets import load_dataset
import os
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# get hf token from environment variable
hf_token = os.getenv("HF_TOKEN")

# hf repos
hf_repo = "jmcinern/"
model = "qwen3-8B-cpt-sft-awq"
dataset= "dpo_dataset_ga"

# load model, tokenizer, dataset
model = AutoModelForCausalLM.from_pretrained(hf_repo+model, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hf_repo+model, trust_remote_code=True)
train_dataset = load_dataset(hf_repo+dataset, split="train")

# setup DPO training
training_args = DPOConfig(output_dir="qomhra-8B-awq-dpo")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()