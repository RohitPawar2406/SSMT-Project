from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")
import os
import torch
import gc

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

os.environ["HF_HOME"] = "/scratch/monica"
os.environ["HF_DATASETS_CACHE"] = "/scratch/monica/hf_cache"   # (optional, you can keep this if needed)
os.environ["HF_DATASETS_TEMP"] = "/scratch/monica/datasets_temp"
from huggingface_hub import login

login("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")
# Set path to your fine-tuned model directory
checkpoint_path = "/home2/surtani.monica/checkpoint-13000"
base_model = "openai/whisper-small"  # Change this if you fine-tuned a different base

# Download and save missing feature extractor (includes preprocessor_config.json)
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
feature_extractor.save_pretrained(checkpoint_path)

# Optional: ensure tokenizer files are also up-to-date (if needed)
# tokenizer = WhisperTokenizer.from_pretrained(base_model)
# tokenizer.save_pretrained(checkpoint_path)

# Optional: combine feature extractor + tokenizer into processor
# processor = WhisperProcessor.from_pretrained(base_model)
# processor.save_pretrained(checkpoint_path)

