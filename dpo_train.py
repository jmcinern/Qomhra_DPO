from datasets import load_dataset
import os
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer
from sklearn.model_selection import train_test_split

# get hf token from environment variable
hf_token = os.getenv("HF_TOKEN")

PatchDPOTrainer()
# hf repos
hf_repo = "jmcinern/"
model_name = "qwen3-8B-cpt-sft-awq"
dataset= "dpo_dataset_ga"

# load model, tokenizer, dataseta
model, tokenizer = FastLanguageModel.from_pretrained(hf_repo+model_name, trust_remote_code=True)

# PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r = 256, # scales performance gains with rank stabilized LoRA. 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 256,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = True,  # rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
# load split
full_dataset = load_dataset(hf_repo+dataset, split="train")
train_data, test_data = train_test_split(full_dataset, test_size=0.1, random_state=42)
# setup DPO training
training_args = DPOConfig(
    output_dir="qomhra-8B-awq-dpo", 
    data_collator = "DataCollatorForPreference", #default
    bf16=True, # unsloth
    per_device_train_batch_size=1, # OOM
    num_train_epochs=3,
    learning_rate=5e-7, 
    warmup_ratio = 0.1,
    loss_type="sigmoid", # standard DPO. Bradley-Terry model# (fdefault)
    beta= 0.1 # default is 0.1 but HF showed 0.01 optimal for DPO, candidates shoul be easy to discern and should avoid underfitting.   
)

trainer = DPOTrainer(
    model=model, 
    args=training_args, 
    processing_class=tokenizer, 
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    report_to="wandb"
    )

trainer.train()