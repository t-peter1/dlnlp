import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from src.model import inject_lora

# --- CONFIG ---
MODEL_ID = "gpt2-medium"
WEIGHTS_PATH = "lora_weights_r4_preproccess.pt"
RANK = 4     
ALPHA = 32
NUM_SAMPLES = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model():
    print("Loading metrics and model...")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    model = inject_lora(model, rank=RANK, alpha=ALPHA)
    
    # load Weights
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    dataset = load_dataset("GEM/e2e_nlg", split="validation", trust_remote_code=True)
    dataset = dataset.select(range(NUM_SAMPLES))

    cols = dataset.column_names
    mr_col = "meaning_representation" if "meaning_representation" in cols else "mr"
    ref_col = "references" if "references" in cols else "target"
    
    print(f"DEBUG: Using columns Input='{mr_col}' and Target='{ref_col}'")

    predictions = []
    references = []

    print(f"Generating responses for {NUM_SAMPLES} examples...")

    for i, example in enumerate(tqdm(dataset)):
        mr = example[mr_col]
        
        input_text = f"{mr} ||" 
        
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=40,
                    num_beams=5,            
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=0.8, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

        generated_tokens = output[0][input_length:]
        full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        """
        # the model generates a long response, which BLEU does not like
        if "." in full_text:
            clean_sentence = full_text.split(".")[0] + "."
        else:
            clean_sentence = full_text
        """
        clean_sentence = full_text
            
        predictions.append(clean_sentence)
        
        ref = example[ref_col]
        if isinstance(ref, str):
            references.append([ref])
        else:
            references.append(ref)

        # 3 examples printed
        if i < 3:
            print(f"\n--- Example {i} ---")
            print(f"INPUT: {mr}")
            print(f"PRED:  {clean_sentence}")
            print(f"REF:   {references[-1][0]}")

    print("\n--- FINAL SCORES ---")
    print(f"BLEU:   {bleu.compute(predictions=predictions, references=references)['bleu']:.4f}")
    print(f"ROUGE-L:{rouge.compute(predictions=predictions, references=references)['rougeL']:.4f}")
    print(f"METEOR: {meteor.compute(predictions=predictions, references=references)['meteor']:.4f}")

if __name__ == "__main__":
    evaluate_model()