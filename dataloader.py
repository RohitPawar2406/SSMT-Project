from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, WhisperProcessor, WhisperTokenizer, WhisperConfig, MarianMTModel, MarianTokenizer
import itertools
import torch


class STT_Dataset(Dataset):
    def __init__(self, dataset, cache_dir, speech_processor, tokenizer_name_marian, max_length=512, training=False):
        self.speech_processor = speech_processor
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_name_marian, cache_dir=cache_dir)
        self.max_length = max_length
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_array = sample["chunked_audio_filepath"]["array"]
        sample_rate = sample["chunked_audio_filepath"]["sampling_rate"]
        
        # Process audio
        audio = self.speech_processor(audio_array, sampling_rate=sample_rate, return_tensors="pt").input_features

        # Tokenize text
        src_texts = sample["text"]
        tgt_texts = sample["hi_text"]
        english_inputs = self.tokenizer(src_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)

        hindi_inputs = self.tokenizer(tgt_texts, return_tensors="pt")

        # Padding Hindi Text
        hindi_inputs_ids  = hindi_inputs["input_ids"].squeeze(0)[:-1] # Remoces last eos token
        target_hindi_ids = hindi_inputs["input_ids"].squeeze(0)
        lenght_of_hindi_inputs = hindi_inputs_ids.shape[0]
        hindi_attention = hindi_inputs["attention_mask"].squeeze(0)[:-1] # Removes last eos token
        padding_idx = self.tokenizer.pad_token_id


        if hindi_inputs_ids.shape[0] >= self.max_length:
            hindi_inputs_ids = hindi_inputs_ids[:self.max_length]
            hindi_attention = hindi_attention[:self.max_length]
        else:
            hindi_inputs_ids = torch.concat((hindi_inputs_ids, torch.ones(self.max_length- hindi_inputs_ids.shape[0], dtype=torch.int32) * padding_idx), dim=0)  # [seq_len]
            constant_zero = torch.zeros(self.max_length- lenght_of_hindi_inputs)
            hindi_attention = torch.concatenate((hindi_attention, constant_zero), dim=0)

        hindi_target = target_hindi_ids[1:] # Removes first bos token
        if hindi_target.shape[0] >= self.max_length:
            hindi_target = hindi_target[:self.max_length]
        else:
            hindi_target = torch.concat((hindi_target, torch.ones(self.max_length- hindi_target.shape[0], dtype=torch.int32) * -100), dim=0)


        return {
            "speech_features": audio.squeeze(0), # (80, 3000)
            "english_input_ids": english_inputs["input_ids"].squeeze(0), # (sequence_length)
            "hindi_input_ids": hindi_inputs_ids, # (sequence_length - 1)
            "hindi_target_ids": hindi_target, # (sequence_length - 1)
            "english_attention_mask": english_inputs["attention_mask"].squeeze(0), # (sequence_length)
            "hindi_attention_mask": hindi_attention, # (sequence_length - 1)
        }
    
if __name__ == "__main__":
    dataset_path = "weights/"  # Change this path
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

    model_medium =  "openai/whisper-medium"  
    cache_dir = "weights/"
    mt_model_name = "Helsinki-NLP/opus-mt-en-hi"
    batch_size = 2
    speech_processor = WhisperProcessor.from_pretrained(model_medium, cache_dir=cache_dir)

    dataset = STT_Dataset(filtered_dataset, cache_dir, speech_processor , mt_model_name, max_length=256)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for batch in train_dataloader:
        print(batch["speech_features"].shape)
        print(batch["english_input_ids"].shape)
        print(batch["hindi_input_ids"].shape)
        print(batch["hindi_target_ids"].shape)
        print(batch["english_attention_mask"].shape)
        print(batch["hindi_attention_mask"].shape)
        print("=============================================")

    breakpoint()