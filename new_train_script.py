import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_scheduler, WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel
from datasets import load_from_disk
import wandb
import os
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from dataloader import STT_Dataset
from model import CustomModel
from evaluate import load

from tqdm import tqdm
import time
from evaluate import load
import glob

# Function to Save Checkpoints (Only Main Process)
def save_checkpoint(epoch, model, optimizer, val_loss = None, filename_prefix="checkpoint"):
    # Save every 5 epochs
    if epoch % 5 == 0:
        filename = f"{filename_prefix}_epoch_MT{epoch}.pth"
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        torch.save(
            {
                "epoch": epoch,
                # "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            },
            save_path
        )
        print(f"📦 Checkpoint saved at {save_path}")

        # Clean up old checkpoints (keep only latest 2)
        checkpoints = sorted(
            glob.glob(os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_epoch.pth")),
            key=os.path.getmtime,
            reverse=True
        )
        for old_ckpt in checkpoints[MAX_CHECKPOINTS:]:
            os.remove(old_ckpt)
            print(f"🧹 Removed old checkpoint: {old_ckpt}")


def load_checkpoint(filename="checkpoint.pth"):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed training from checkpoint: {checkpoint_path}")
        return checkpoint["epoch"]
    return 0


def train_one_epoch(train_dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Prepare inputs
        speech_features = batch["speech_features"].to(device)
        english_input_ids = batch["english_input_ids"].to(device)
        hindi_input_ids = batch["hindi_input_ids"].to(device)
        hindi_target_ids = batch["hindi_target_ids"].to(device)
        english_attention_mask = batch["english_attention_mask"].to(device)
        hindi_attention_mask = batch["hindi_attention_mask"].to(device)

        # Forward Pass
        outputs = model(english_input_ids, hindi_input_ids, speech_features , hindi_attention_mask=hindi_attention_mask,english_attention_mask=english_attention_mask)

        loss = criterion(outputs.view(-1, TARGET_VOCAB_SIZE), hindi_target_ids.view(-1))

        # Backward Pass with Gradient Accumulation
        loss.backward()
        # if (step + 1) % grad_step == 0:  # Gradient accumulation step
        optimizer.step()
            # scheduler.step()

        total_loss += loss.item()

        
        writer.add_scalar("Loss/train_loss_per_step", loss.item(), step)
        wandb.log({"train_loss_per_step": loss.item()})

    return total_loss / len(train_dataloader)


if __name__ == "__main__":
    #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
    # Load metrics globally (optional caching)
    # bertscore_metric = load("bertscore")
    #comet_metric = load("comet") 

    # mention WER, COMETloss = criterion(outputs.view(-1, TARGET_VOCAB_SIZE), target_ids.view(-1))
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./tensorboard_logs"
    NUM_EPOCHS = 6
    MAX_CHECKPOINTS = 2
    max_len = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_training = False  # Set to True to resume training from the last checkpoint
    batch_size = 16
    dataset_path = "weights/"  # Change this path
    model_medium =  "openai/whisper-medium"  
    cache_dir = "weights/"
    mt_model_name = "Helsinki-NLP/opus-mt-en-hi"
    speech_processor = WhisperProcessor.from_pretrained(model_medium, cache_dir=cache_dir)
    Marian_model = MarianMTModel.from_pretrained(mt_model_name, add_cross_attention=False, cache_dir=cache_dir)
    Whisper_model = WhisperForConditionalGeneration.from_pretrained(model_medium, cache_dir=cache_dir)

    wandb.init(project="whisper-speech-to-text")
    writer = SummaryWriter(LOG_DIR)

    ########## dataset and dataloader ##########
    raw_dataset = load_from_disk(f"{dataset_path}word-poject")

    #raw_dataset = load_from_disk(f"{dataset_path}ugce")
    len_of_raw_datatset = len(raw_dataset)
    print("Length of raw dataset:", len_of_raw_datatset)

    # List of indices you want to remove
    indices_to_delete = [29234, 29235, 29236, 29237, 29238, 29239, 29240, 29241, 29242, 29243, 29244, 29245, 29246, 29247, 29248, 29249, 29250, 29251, 29252, 29253, 29254, 29255, 29256, 29257, 29258, 29259, 29260, 29261, 29262, 29263, 29264, 29265, 29266, 29267, 29268, 29269, 29270, 29271, 29272, 29273, 29274, 29275, 29276, 29277, 29278, 29279, 29280, 29281, 29282, 29283, 29284, 29285, 29286]

    # Create a set for faster lookup
    indices_to_delete_set = set(indices_to_delete)

    # Apply the filter
    filtered_dataset = raw_dataset.filter(
        lambda example, idx: idx not in indices_to_delete_set,
        with_indices=True
    )
    filtered_dataset_len = len(filtered_dataset)
    print("Length of Filtered dataset: ", filtered_dataset_len)

    # Initialize dataset and dataloader
    dataset = STT_Dataset(filtered_dataset, cache_dir, speech_processor , mt_model_name, max_length=max_len)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    ################################################################
    TARGET_VOCAB_SIZE = 61950

    # Initialize the model
    model = CustomModel(Marian_model, Whisper_model)
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    criterion = nn.CrossEntropyLoss()

    # Main Training Loop
    if resume_training:
        start_epoch = load_checkpoint()  # Resume if checkpoint exists
    else:
        start_epoch = 0

    
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(train_dataloader, model, optimizer, criterion)

        wandb.log({"train_loss": train_loss, 
                   "epoch": epoch})

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)

        elapsed_minutes = (time.time() - start_time) / 60
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} |  Time: {elapsed_minutes:.2f} min")
        print("=" * 120)

    
    writer.close()
    wandb.finish()

