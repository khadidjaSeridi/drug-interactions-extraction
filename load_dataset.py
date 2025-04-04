#from datasets import load_dataset
import os



from datasets import load_dataset
import random
import json

# Load dataset
dataset_name = "SkyHuReal/DrugBank-Alpaca"
dataset = load_dataset(dataset_name, split="train")

# Function to preprocess data into model-friendly format
def preprocess_data(example):
    """Convert instruction + input into a single prompt, keep output as response"""
    prompt = f"\n{example['instruction']} {example['input']}\n\n Response:"
    response = example["output"].strip()

    return {"prompt": prompt, "response": response}

def split_list(data, split_ratio=0.8):
    """Splits a list into two random subsets with given ratio."""
    size = int(len(data) * split_ratio)
    random.shuffle(data)
    return data[:size], data[size:]

# Apply preprocessing
processed_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names)

dataset_list = processed_dataset.to_list()

# Split dataset into train and test sets
train_data, test_data = split_list(dataset_list)

# Save the formatted dataset to JSON
output_file = os.path.join(os.path.dirname(__file__),"data","preprocessed_drug_relation_extraction_train.json")
with open(output_file, "w") as f:
    json.dump(train_data, f, indent=2)

output_file = os.path.join(os.path.dirname(__file__),"data","preprocessed_drug_relation_extraction_test.json")
with open(output_file, "w") as f:
    json.dump(test_data, f, indent=2)

