import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import evaluate
from tqdm import tqdm

from src.adalora_model import inject_adalora


# --- CONFIG ---
MODEL_ID = "gpt2-medium"
CKPT_PATH = "adalora_full_checkpoint.pt"  # default to the full-run AdaLoRA checkpoint
RANK_INIT = 4
ALPHA = 32
NUM_SAMPLES = 100

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model():
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("meta", {})
    else:
        state_dict = checkpoint
        meta = {}

    rank = meta.get("rank_init", RANK_INIT)
    alpha = meta.get("alpha", ALPHA)

    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

    from src.adalora_model import AdaLoRAConv1D, inject_adalora
    model, _ = inject_adalora(model, rank=rank, alpha=alpha)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, AdaLoRAConv1D):
                lam_key = f"{name}.lambda_vals"
                if lam_key in state_dict:
                    checkpoint_rank = state_dict[lam_key].shape[0]
                    if checkpoint_rank != module.rank:
                        in_features = module.original_layer.weight.shape[0]
                        out_features = module.original_layer.weight.shape[1]

                        module.Q = torch.nn.Parameter(torch.empty(checkpoint_rank, in_features))
                        module.P = torch.nn.Parameter(torch.empty(out_features, checkpoint_rank))
                        module.lambda_vals = torch.nn.Parameter(torch.empty(checkpoint_rank))
                        module.rank = checkpoint_rank
                        if "rank_mask" in module._buffers:
                            del module._buffers["rank_mask"]
                        module.register_buffer("rank_mask", torch.ones(checkpoint_rank))
                        module.scaling = module.alpha / checkpoint_rank

    model.load_state_dict(state_dict, strict=False)

    active = 0
    total = 0
    for name, b in model.named_buffers():
        if "rank_mask" in name:
            active += int((b > 0).sum().item())
            total += b.numel()
    print(f"[eval] active mask directions: {active}/{total}")
    model.to(device)
    model.eval()

    dataset = load_dataset("GEM/e2e_nlg", split="validation", trust_remote_code=True)
    dataset = dataset.select(range(NUM_SAMPLES))

    cols = dataset.column_names
    mr_col = "meaning_representation" if "meaning_representation" in cols else "mr"
    ref_col = "references" if "references" in cols else "target"

    predictions = []
    references = []

    for i, example in enumerate(tqdm(dataset)):
        mr = example[mr_col]
        input_text = f"{mr} ||"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

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

        generated_tokens = output[0][input_len:]
        full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        clean_sentence = full_text

        predictions.append(clean_sentence)
        ref = example[ref_col]
        references.append([ref] if isinstance(ref, str) else ref)

        if i < 3:
            print(f"\n--- Example {i} ---")
            print(f"INPUT: {mr}")
            print(f"PRED:  {clean_sentence}")
            print(f"REF:   {references[-1][0]}")

    print("\n--- FINAL SCORES ---")
    print(f"BLEU:   {bleu.compute(predictions=predictions, references=references)['bleu']:.4f}")
    print(
        f"ROUGE-L:{rouge.compute(predictions=predictions, references=references)['rougeL']:.4f}"
    )
    print(f"METEOR: {meteor.compute(predictions=predictions, references=references)['meteor']:.4f}")


if __name__ == "__main__":
    evaluate_model()
