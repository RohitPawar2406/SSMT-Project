import os
import glob
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import get_scheduler, WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel
from datasets import load_from_disk
import wandb

from dataloader import STT_Dataset
from model import CustomModel
from evaluate import load
from tqdm import tqdm

# ---------------- DDP SETUP ----------------
def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ---------------- CHECKPOINTING ----------------
def save_checkpoint(epoch, model, optimizer, filename_prefix="checkpoint"):
    # only rank 0 writes
    if dist.get_rank() != 0:
        return
    if epoch % 5 == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        filename = f"{filename_prefix}_epoch_MT{epoch}.pth"
        save_path = os.path.join(CHECKPOINT_DIR, filename)
        # unwrap DDP
        state_dict = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)
        print(f"📦 Checkpoint saved at {save_path}")

        # clean up old
        pattern = os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_epoch_MT*.pth")
        ckpts = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for old in ckpts[MAX_CHECKPOINTS:]:
            os.remove(old)
            print(f"🧹 Removed old checkpoint: {old}")

def load_checkpoint(model, optimizer, filename=None):
    # pick latest if none specified
    if filename is None:
        pattern = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_MT*.pth")
        ckpts = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not ckpts:
            return 0
        filename = ckpts[0]
    map_loc = {'cuda:%d' % 0: f'cuda:{LOCAL_RANK}'}
    ckpt = torch.load(filename, map_location=map_loc)
    # unwrap DDP
    target = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    target.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[RANK {dist.get_rank()}] Loaded checkpoint: {filename}")
    return ckpt["epoch"]

# ---------------- TRAINING ----------------
def train_one_epoch(train_dataloader, model, optimizer, criterion, epoch):
    model.train()
    # set epoch for sampler shuffle
    train_dataloader.sampler.set_epoch(epoch)
    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # move inputs to local GPU
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        outputs = model(
            batch["english_input_ids"],
            batch["hindi_input_ids"],
            batch["speech_features"],
            hindi_attention_mask=batch["hindi_attention_mask"],
            english_attention_mask=batch["english_attention_mask"]
        )

        loss = criterion(outputs.view(-1, TARGET_VOCAB_SIZE),
                         batch["hindi_target_ids"].view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if RANK == 0:
            writer.add_scalar("Loss/train_loss_per_step", loss.item(), step)
            wandb.log({"train_loss_per_step": loss.item()})

    return total_loss / len(train_dataloader)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    LOCAL_RANK = setup_ddp()
    RANK       = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    # directories & hyperparams
    CHECKPOINT_DIR   = "./checkpoints"
    LOG_DIR          = "./tensorboard_logs"
    NUM_EPOCHS       = 6
    MAX_CHECKPOINTS  = 2
    max_len          = 256
    batch_size       = 16
    dataset_path     = "weights/"
    model_medium     = "openai/whisper-medium"
    cache_dir        = "weights/"
    mt_model_name    = "Helsinki-NLP/opus-mt-en-hi"
    TARGET_VOCAB_SIZE = 61950

    # only rank0 logs
    if RANK == 0:
        wandb.init(project="whisper-speech-to-text")
        writer = SummaryWriter(LOG_DIR)

    # load metrics
    bertscore_metric = load("bertscore")

    # data
    raw_dataset = load_from_disk(f"{dataset_path}word-poject")
    indices_to_delete = [29234, 29235, 29236, 29237, 29238, 29239, 29240,
                         29241, 29242, 29243, 29244, 29245, 29246, 29247,
                         29248, 29249, 29250, 29251, 29252, 29253, 29254,
                         29255, 29256, 29257, 29258, 29259, 29260, 29261,
                         29262, 29263, 29264, 29265, 29266, 29267, 29268,
                         29269, 29270, 29271, 29272, 29273, 29274, 29275,
                         29276, 29277, 29278, 29279, 29280, 29281, 29282,
                         29283, 29284, 29285, 29286]
    filtered_dataset = raw_dataset.filter(
        lambda ex, idx: idx not in set(indices_to_delete),
        with_indices=True
    )

    dataset = STT_Dataset(
        filtered_dataset,
        cache_dir,
        WhisperProcessor.from_pretrained(model_medium, cache_dir=cache_dir),
        mt_model_name,
        max_length=max_len
    )

    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # model
    marian  = MarianMTModel.from_pretrained(mt_model_name, add_cross_attention=False, cache_dir=cache_dir)
    whisper = WhisperForConditionalGeneration.from_pretrained(model_medium, cache_dir=cache_dir)
    model   = CustomModel(marian, whisper).cuda()

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        find_unused_parameters=False
    )

    # optimizer & loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # optionally resume
    start_epoch = 0
    if os.getenv("RESUME", "0") == "1":
        start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(train_dataloader, model, optimizer, criterion, epoch)

        if RANK == 0:
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            save_checkpoint(epoch, model, optimizer)

        elapsed = (time.time() - start_time) / 60
        if RANK == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Time: {elapsed:.2f} min")
            print("=" * 120)

    if RANK == 0:
        writer.close()
        wandb.finish()

    cleanup_ddp()
