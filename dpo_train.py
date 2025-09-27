from datasets import load_dataset
import os
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer
import wandb
from transformers import EarlyStoppingCallback

# DPO
# 1. "Improvement", Scores vs Reference: s(y) = log πθ​(y|x) - log π_ref(y|x)
# 2.  "Margin" (bigger better): m = s(y_chosen) - s(y_reject)
# 3.  Scale margin by beta: β * m 
# NB: lower beta => smaller margin 
# 4. Loss: L = -log σ(β * m) <- "proba the winner beats the loser"
# NB bigger margin => proba ~ 1 => loss ~ 0 
# So smaller beta reduces improvement margin giving higher loss and thus more aggressive drift.

# beta scales the margin between win and loss. 
# Smaller beta means, margin are scaled down more agressively, increased loss.  
betas = [0.01, 0.1, 0.5]
# get hf token from environment variable
hf_token = os.getenv("HF_TOKEN")

PatchDPOTrainer()
# hf repos
hf_repo = "jmcinern/"
model_name = "qwen3-8B-cpt-sft-awq"
dataset= "dpo_dataset_ga"

# load split
full_dataset = load_dataset(hf_repo+dataset, use_auth_token=hf_token)
full_dataset = full_dataset['train']
dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset_split["train"]
test_data = dataset_split["test"]

for i, beta in enumerate(betas):
    wandb.init(project="qomhra-dpo-beta", name=f"FULL_RUN_dpo-{model_name}-{i}-beta-{beta}", reinit=True)

    # load model, tokenizer, dataseta
    model, tokenizer = FastLanguageModel.from_pretrained(hf_repo+model_name, trust_remote_code=True)

    # PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r = 256, # scales performance gains with rank stabilized LoRA. 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 256,
        lora_dropout = 0.0, # don't inject noise during DPO 
        bias = "none",    # Currently only supports bias = "none"
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        use_rslora = True,  # rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    # setup DPO training
    training_args = DPOConfig(
        output_dir=f"qomhra-8B-awq-dpo-beta-{beta}", 
        fp16=True, # bf16 not supported by v100
        per_device_train_batch_size=1, # OOM
        num_train_epochs=3,
        learning_rate=5e-7, 
        warmup_ratio = 0.1,
        loss_type="sigmoid", # standard DPO. Bradley-Terry model# (fdefault)
        beta=beta, # default, open question.
        report_to="wandb",
        logging_steps=5, # default is 10
        eval_on_start=True,

        # overfitting mitigation
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0, # clipping
        eval_strategy="steps",       
        eval_steps=25,               
        save_strategy="steps",      
        save_steps=25,               
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        greater_is_better=False, 
        per_device_eval_batch_size=1,
        save_total_limit=3
    )

    #early stopping
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,      # Stop after 2 evals without improvement
        early_stopping_threshold=0.001  # Minimum improvement threshold
    )
    trainer = DPOTrainer(
        model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_data,
        eval_dataset=test_data,
        callbacks=[early_stopping_callback]
        )

    trainer.train()
    # fresh wandb run for each beta
    wandb.finish()