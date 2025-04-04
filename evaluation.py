import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
import datasets
import evaluate  
from peft import PeftModel
import json

# Define model and tokenizer paths
MODEL_NAME = "bigscience/bloom-560m"  # Base model
FINETUNED_MODEL_PATH = "output"  # Path where  LoRA model is saved

# Load base model and apply LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load evaluation dataset
raw_data =  json.loads(open("data/preprocessed_drug_relation_extraction_test.json").read())

dataset = Dataset.from_list(raw_data)

generation_config = GenerationConfig(**{
   # "temperature": 0.1,    # Minimal randomness
   # "top_k": 5,            # Only consider the top 5 tokens
   # "top_p": 0.3,          # Limit token choices to 30% probability mass
    "max_new_tokens": 20,  # Avoid overly long responses
    "repetition_penalty": 1.2  # Prevents repeating input text
}) # Create a GenerationConfig object

# Define evaluation function
def evaluate_model(model, tokenizer, dataset, num_samples=50):
    """
    Evaluates the fine-tuned model on a subset of the evaluation dataset.
    Compares generated outputs against reference answers.
    """
    model.eval()
    metric_bleu = evaluate.load("bleu")
  #  metric_rouge = evaluate.load("rouge")

    predictions, references = [], []

    for i, sample in enumerate(dataset.select(range(num_samples))):
        prompt = sample["prompt"].strip()
        expected_output = sample["response"].strip()

        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=generation_config)
        
        generated_text = tokenizer.decode(output_ids[0])#, skip_special_tokens=True)

        # Store results
        predictions.append(generated_text)
        references.append([expected_output])  # BLEU expects a list of references

        print(f"Sample {i+1}/{num_samples}")
        print(f"Prompt: {prompt}")
        #print(f"Input: {input_text}")
        print(f"Expected Output: {expected_output}")
        print(f"Generated Output: {generated_text}")
        print("-" * 80)

    # Compute metrics
    bleu_score = metric_bleu.compute(predictions=predictions, references=references)["bleu"]
    #rouge_score = metric_rouge.compute(predictions=predictions, references=references)

    print("\nEvaluation Results:")
    print(f"BLEU Score: {bleu_score:.4f}")
   # print(f"ROUGE Score: {rouge_score}")

# Run evaluation
evaluate_model(model, tokenizer, dataset)
