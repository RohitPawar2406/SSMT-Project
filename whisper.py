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

from tqdm import tqdm
tqdm.pandas()
from datasets import load_dataset, DatasetDict
############################################################
print("loading UGCE-Resources dataset1")
bhasaanuvaad = load_dataset("ai4bharat/UGCE-Resources", "en2indic", split="en2indic", cache_dir="/scratch/monica")
dataset1 = load_dataset("ai4bharat/UGCE-Resources", "en2indic")
print("loading WordProject dataset2")
bhasaanuvaad = load_dataset("ai4bharat/WordProject", "en2indic", split="en2indic", cache_dir="/scratch/monica")
dataset2 = load_dataset("ai4bharat/WordProject", "en2indic")
#print(dataset)
#print(dataset["en2indic"][0]["hi_text"])

############################################################
print("featureextractor and tokenizer")
from transformers import WhisperFeatureExtractor
torch.cuda.empty_cache()
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", cache_dir="/scratch/monica")
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="translate", cache_dir="scratch/monica")

#############################################################
print("tokening dataset1")
input_str = dataset1['en2indic'][0]['hi_text']
print(input_str)
# Assuming tokenizer is already defined
final_dataset1 = dataset1
input_str = final_dataset1['en2indic'][0]['hi_text']  # Use final_dataset instead of dataset
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
###############################################################
print("tokening dataset2")
input_str = dataset2['en2indic'][0]['hi_text']
print(input_str)
# Assuming tokenizer is already defined
final_dataset2 = dataset2
input_str = final_dataset2['en2indic'][0]['hi_text']  # Use final_dataset instead of dataset
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
#################################################################
print("Processor")
# from transformers import WhisperProcessor

# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="translate", cache_dir="/scratch/monica")
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",language="Hindi", task="translate",
    cache_dir="/scratch/monica"
)
##################################################################
print("total dataset1 for training")
final_dataset1 = final_dataset1.remove_columns(
    [
        col for col in ['pa_text', 'pa_mining_score', 'or_text', 'or_mining_score', 'ur_text', 'ur_mining_score', 'text', 'pred_text', 'audio_filepath', 'start_time', 'duration', 'alignment_score', 'mr_text', 'kn_text', 'bn_mining_score', 'ta_text', 'ml_text', 'gu_mining_score', 'ta_mining_score', 'bn_text', 'kn_mining_score', 'hi_mining_score', 'gu_text', 'ml_mining_score', 'te_mining_score', 'mr_mining_score', 'te_text']
        if col in final_dataset1["en2indic"].column_names  # Only remove if the column exists
    ]
)
print(final_dataset1)
indices_to_delete = set(range(22000, 163121))
#indices_to_delete = set(range(8000, 163121))
# Update each split in the DatasetDict
for split in final_dataset1.keys():
    # Select the indices to keep for this split
    indices_to_keep = [i for i in range(len(final_dataset1[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
    final_dataset1[split] = final_dataset1[split].select(indices_to_keep)
    # Print the length of the modified split
    print(f"Length of {split}: {len(final_dataset1[split])}")
###################################################################
print("total dataset2 for training")
indices_to_delete = set(range(8000, 163121))
#indices_to_delete = set(range(8000, 163121))
# Update each split in the DatasetDict
for split in final_dataset2.keys():
    # Select the indices to keep for this split
    indices_to_keep = [i for i in range(len(final_dataset2[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
    final_dataset2[split] = final_dataset2[split].select(indices_to_keep)
    # Print the length of the modified split
    print(f"Length of {split}: {len(final_dataset2[split])}")    
##################################################################    
print("merge final_dataset")
from datasets import DatasetDict, concatenate_datasets

# First, print the existing datasets (optional for checking)
print(final_dataset1)
print(final_dataset2)

# Create an empty DatasetDict to store the merged datasets
final_dataset = DatasetDict()

# Loop over the splits (e.g., "train", "test", etc.)
for split in final_dataset1.keys():
    # Concatenate the corresponding splits from final_dataset1 and final_dataset2
    merged_split = concatenate_datasets([final_dataset1[split], final_dataset2[split]])
    # Store in final_dataset
    final_dataset[split] = merged_split
    # Print the length of the merged split
    print(f"Length of merged {split}: {len(final_dataset[split])}")

# Now final_dataset contains the merged data!
###################################################################
from datasets import Audio

final_dataset = final_dataset.cast_column("chunked_audio_filepath", Audio(sampling_rate=16000))
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    # print(batch)
    audio = batch["chunked_audio_filepath"]

    # compute log-Mel input features from input audio array

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["hi_text"]).input_ids
    return batch
counter = {"count": 0}

# Define prepare_dataset function
def prepare_dataset(batch):
    counter["count"] += 1
    if counter["count"] % 5000 == 0:
        print(f"Processed {counter['count']} examples")

    # load and resample audio data
    audio = batch["chunked_audio_filepath"]

    # compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["hi_text"]).input_ids

    return batch

# Apply mapping
final_dataset = final_dataset.map(
    prepare_dataset,
    remove_columns=final_dataset["en2indic"].column_names,  # corrected: use ["en2indic"]
    num_proc=1
)
print("final_dataset after map")
print(final_dataset)
#########################################################################
#indices_to_delete = set(range(38900, 39000))
# Update each split in the DatasetDict
#for split in final_dataset.keys():
    # Select the indices to keep for this split
 #   indices_to_keep = [i for i in range(len(final_dataset[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
#    final_dataset[split] = final_dataset[split].select(indices_to_keep)
    # Print the length of the modified split
#    print(f"Length of {split}: {len(final_dataset[split])}")
print("eval datasets preparation starts")
print(dataset1)
print(dataset2)
###########################################################################
# Assuming tokenizer is already defined
print("eval_dataset1 tokening")
eval_dataset1 = dataset1
input_str = eval_dataset1['en2indic'][0]['hi_text']  # Use final_dataset instead of dataset
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

eval_dataset1 = eval_dataset1.remove_columns(
    [
        col for col in ['pa_text', 'pa_mining_score', 'or_text', 'or_mining_score', 'ur_text', 'ur_mining_score', 'text', 'pred_text', 'audio_filepath', 'start_time', 'duration', 'alignment_score', 'mr_text', 'kn_text', 'bn_mining_score', 'ta_text', 'ml_text', 'gu_mining_score', 'ta_mining_score', 'bn_text', 'kn_mining_score', 'hi_mining_score', 'gu_text', 'ml_mining_score', 'te_mining_score', 'mr_mining_score', 'te_text']
        if col in eval_dataset1["en2indic"].column_names  # Only remove if the column exists
    ]
)
print(eval_dataset1)

indices_to_delete = set(range(1, 22000))
#indices_to_delete = set(range(1, 8000))
# Update each split in the DatasetDict
for split in eval_dataset1.keys():
    # Select the indices to keep for this split
    indices_to_keep = [i for i in range(len(eval_dataset1[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
    eval_dataset1[split] = eval_dataset1[split].select(indices_to_keep)
    # Print the length of the modified split
    print(f"Length of {split}: {len(eval_dataset1[split])}")
indices_to_delete = set(range(3000, 1700000
                             ))
# Update each split in the DatasetDict
for split in eval_dataset1.keys():
    # Select the indices to keep for this split
    indices_to_keep = [i for i in range(len(eval_dataset1[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
    eval_dataset1[split] = eval_dataset1[split].select(indices_to_keep)
    # Print the length of the modified split
    print(f"Length of {split}: {len(eval_dataset1[split])}")
##########################################################################
print("eval_dataset2 preparing")
# Assuming tokenizer is already defined
print("eval_dataset2 tokening")
bhasaanuvaad = load_dataset("ai4bharat/WordProject", "en2indic", split="en2indic", cache_dir="/scratch/monica")
dataset2 = load_dataset("ai4bharat/WordProject", "en2indic")
eval_dataset2 = dataset2
print(eval_dataset2)
input_str = eval_dataset2['en2indic'][0]['hi_text']  # Use final_1 instead of 1
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")
print(eval_dataset2)
eval_dataset2 = eval_dataset2.remove_columns(
    [
        col for col in ['pa_text', 'pa_mining_score', 'or_text', 'or_mining_score', 'ur_text', 'ur_mining_score', 'text', 'pred_text', 'audio_filepath', 'start_time', 'duration', 'alignment_score', 'mr_text', 'kn_text', 'bn_mining_score', 'ta_text', 'ml_text', 'gu_mining_score', 'ta_mining_score', 'bn_text', 'kn_mining_score', 'hi_mining_score', 'gu_text', 'ml_mining_score', 'te_mining_score', 'mr_mining_score', 'te_text']
        if col in eval_dataset2["en2indic"].column_names  # Only remove if the column exists
    ]
)
print(eval_dataset2)

indices_to_delete = set(range(1, 8000))
#indices_to_delete = set(range(1, 8000))
# Update each split in the DatasetDict
for split in eval_dataset2.keys():
    # Select the indices to keep for this split
    indices_to_keep = [i for i in range(len(eval_dataset2[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
    eval_dataset2[split] = eval_dataset2[split].select(indices_to_keep)
    # Print the length of the modified split
    print("first indice eval2 delete")
    print(f"Length of {split}: {len(eval_dataset2[split])}")
indices_to_delete = set(range(1500, 170000
                             ))
# Update each split in the DatasetDict
for split in eval_dataset2.keys():
    # Select the indices to keep for this split
    indices_to_keep = [i for i in range(len(eval_dataset2[split])) if i not in indices_to_delete]
    # Update the split with the selected indices
    eval_dataset2[split] = eval_dataset2[split].select(indices_to_keep)
    # Print the length of the modified split
    print("second indice eval2 delete")
    print(f"Length of {split}: {len(eval_dataset2[split])}")
##################################################################
print("merge eval_dataset")

from datasets import DatasetDict, concatenate_datasets

# First, print the existing datasets (optional for checking)
print(eval_dataset1)
print(eval_dataset2)

# Create an empty DatasetDict to store the merged datasets
eval_dataset = DatasetDict()

# Loop over the splits (e.g., "train", "test", etc.)
for split in eval_dataset1.keys():
    # Concatenate the corresponding splits from eval_dataset1 and eval_dataset2
    merged_split = concatenate_datasets([eval_dataset1[split], eval_dataset2[split]])
    # Store in eval_dataset
    eval_dataset[split] = merged_split
    # Print the length of the merged split
    print(f"Length of merged {split}: {len(eval_dataset[split])}")

# Now eval_dataset contains the merged data
##########################################################################
from datasets import Audio

eval_dataset = eval_dataset.cast_column("chunked_audio_filepath", Audio(sampling_rate=16000))
counter = {"count": 0}

# Define prepare_dataset function
def prepare_dataset(batch):
    counter["count"] += 1
    if counter["count"] % 1000 == 0:
        print(f"Processed {counter['count']} examples")

    # load and resample audio data
    audio = batch["chunked_audio_filepath"]

    # compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["hi_text"]).input_ids

    return batch

# Apply mapping
eval_dataset = eval_dataset.map(
    prepare_dataset,
    remove_columns=eval_dataset["en2indic"].column_names,  # corrected: use ["en2indic"]
    num_proc=1
)

print("eval after map")

print(eval_dataset)
##############################################################
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    cache_dir="/scratch/monica"  
)
############
model.gradient_checkpointing_disable() 
############
model.generation_config.language = "hindi"
model.generation_config.task = "translate"
model.generation_config.forced_decoder_ids = None

max_label_length = model.config.max_length
def filter_labels(labels):
    """Filter label sequences longer than max length"""
    return len(labels) < max_label_length
final_dataset = final_dataset.filter(filter_labels, input_columns=["labels"])
eval_dataset = eval_dataset.filter(filter_labels, input_columns=["labels"])

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]


        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
import evaluate
import os
import torch
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq, WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load SacreBLEU
sacrebleu_metric = evaluate.load("sacrebleu")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # SacreBLEU expects references as list of list
    references = [[ref] for ref in label_str]

    # Compute SacreBLEU
    bleu_score = 100 * sacrebleu_metric.compute(predictions=pred_str, references=references)["score"]

    return {
        "sacrebleu": bleu_score,
    }

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/scratch/monica/whisper-small-hi-30k",
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=25,
    gradient_checkpointing=False,
    fp16=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    greater_is_better=True,
    logging_steps=100,
    logging_first_step=True,
    logging_dir="/scratch/monica/logs",
    report_to=["tensorboard"],
    predict_with_generate=True,
    push_to_hub=False,
    optim="adamw_torch_fused",
    fp16_full_eval=True,
    max_grad_norm=1.0,
)

# Custom callback to save model on improved SacreBLEU score
from transformers import TrainerCallback

class SaveBestBLEUCallback(TrainerCallback):
    def __init__(self):
        self.best_bleu = -1

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        current_bleu = metrics.get("sacrebleu", None)
        if current_bleu is not None and current_bleu > self.best_bleu:
            self.best_bleu = current_bleu
            epoch = int(state.epoch) if state.epoch is not None else 0
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}-BLEU{current_bleu:.2f}")
            print(f"\n✨ New best SacreBLEU = {current_bleu:.2f}, saving model to: {save_path}")
            model.save_pretrained(save_path)
            processor.tokenizer.save_pretrained(save_path)

# Prepare datasets
train_dataset = final_dataset["en2indic"]
eval_dataset = eval_dataset["en2indic"]

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestBLEUCallback()],
)

# Start training
trainer.train()

# Save final model
print("Saving final model...")
trainer.save_model(training_args.output_dir)
trainer.save_state()
model.save_pretrained(training_args.output_dir)
processor.tokenizer.save_pretrained(training_args.output_dir)

print("✅ Training completed and model saved.")


