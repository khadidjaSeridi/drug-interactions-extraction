import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import json


model_name = "bigscience/bloom-560m"

model = AutoModelForCausalLM.from_pretrained(
   # load_in_8bits=True,
    model_name,
    device_map = "auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Freezing the original weights 

for param in model.parameters():
    param.requires_grad = False # feeze the model
    
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()   #reduce number of stored activations ?
    model.enable_input_require_grads()

class CastoutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

model.lm_head = CastoutputToFloat(model.lm_head)

# Setting up the LoRa adapter

# def print_trainable_parameters(model):
#     """
#     prints the number of trainable params on the model
#     """

#     trainable_params = 0
#     all_params = 0
#     for _,param in model.named_parameters():
#         all_params += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
    
#     print("trainables params : "+ str(trainable_params) +", all_params :" + str(all_params) +", rate of tranabile : " + str((100*trainable_params/all_params)))

config = LoraConfig(
    r =16, #attention heads
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model,config)

# print_trainable_parameters(model) 

#raw_data = load_dataset("SkyHuReal/DrugBank-Alpaca", trust_remote_code=True, split="train")
#raw_data =  json.load(open("data/preprocessed_drug_relation_extraction_train.json"))

with open("data/preprocessed_drug_relation_extraction_train.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)  # mapping is a list of strings


dataset = Dataset.from_list([{"text": text} for text in mapping])

# Function to tokenize the dataset
def tokenize_function(example):
    return tokenizer(example, truncation=True, padding="max_length", max_length=512)


# Apply tokenization
tokenized_data = dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)

# Training the model

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=transformers.TrainingArguments(
        gradient_accumulation_steps=1,
        per_device_train_batch_size=3,
        warmup_steps=100,
        max_steps=200, 
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="output"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
model.config.use_cache = False # re-enable for inference 
trainer.train()
model.save_pretrained("output")



