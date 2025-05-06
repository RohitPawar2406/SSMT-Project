from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
import sacrebleu
import evaluate
import os

# Setup
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Cache paths
os.environ["HF_HOME"] = "/scratch/monica"
os.environ["HF_DATASETS_CACHE"] = "/scratch/monica/hf_cache"
os.environ["HF_DATASETS_TEMP"] = "/scratch/monica/datasets_temp"

# Load dataset
dataset = load_dataset("ai4bharat/WordProject", "en2indic", cache_dir="/scratch/monica")
subset_dataset = dataset["en2indic"].select(range(9501, 10302))
ground_truth_texts = [row["hi_text"] for row in subset_dataset]

# Ensure audio column is in the right format
subset_dataset = subset_dataset.cast_column("chunked_audio_filepath", Audio(sampling_rate=16000))

# Load processor and model
checkpoint_path = "/home2/surtani.monica/checkpoint-13000"
processor = WhisperProcessor.from_pretrained(checkpoint_path, language="Hindi", task="translate", cache_dir="/scratch/monica")
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path, cache_dir="/scratch/monica").to(device)

# Preprocess function
def preprocess_function(batch):
    audio = batch["chunked_audio_filepath"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = batch["hi_text"]
    return batch

subset_dataset = subset_dataset.map(preprocess_function, remove_columns=subset_dataset.column_names)

# Collate function
def collate_fn(batch):
    input_features = pad_sequence([torch.tensor(example["input_features"]) for example in batch], batch_first=True)
    labels = [example["labels"] for example in batch]
    return {"input_features": input_features, "labels": labels}

dataloader = DataLoader(subset_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Inference
predicted_texts = []
all_ground_truths = []

model.eval()
with torch.no_grad():
    for batch in dataloader:
        # input_features = batch["input_features"].to(device)
        # with autocast():
        #     generated_ids = model.generate(input_features, max_length=128)
        # preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # predicted_texts.extend(preds)
        all_ground_truths.extend(batch["labels"])

with open("predicted_model_1000.txt", "r") as f:
    lines = f.readlines()

predicted_texts = lines
# Remove the newline characters if needed
#breakpoint()



# with open("predicted_model_1000.txt", "r") as f:
#     lines = f.readlines()

# Remove the newline characters if needed


# BLEU (SacreBLEU)
from sacrebleu import corpus_bleu, sentence_bleu

sacre_bleu_score = corpus_bleu(predicted_texts, [all_ground_truths]).score

# COMET
from comet import download_model, load_from_checkpoint

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)

comet_data = [{"src": " ", "mt": mt, "ref": ref} for mt, ref in zip(predicted_texts, all_ground_truths)]
comet_score = comet_model.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
# CHRF++
chrf_score = sacrebleu.corpus_chrf(predicted_texts, [all_ground_truths], word_order=2).score  # word_order=2 gives CHRF++

# Save results
with open("metrics_summary_WP_1000.txt", "w", encoding="utf-8") as f:
    f.write(f"SacreBLEU: {sacre_bleu_score:.2f}\n")
    f.write(f"COMET: {comet_score.system_score:.4f}\n")
    f.write(f"CHRF++: {chrf_score:.2f}\n")


# Also save predictions and references
with open("predictions_WP_1000.txt", "w", encoding="utf-8") as f_pred, \
     open("ground_truths_WP_1000.txt", "w", encoding="utf-8") as f_gt, \
     open("sentence_bleu_scores_WP_1000.txt", "w", encoding="utf-8") as f_bleu:

    for pred, gt in zip(predicted_texts, all_ground_truths):
        f_pred.write(pred.strip() + "\n")
        f_gt.write(gt.strip() + "\n")
        score = sentence_bleu(pred.strip(), [gt.strip()])
        f_bleu.write(f"{score.score:.2f}\n")

print("All metrics and outputs saved.")

