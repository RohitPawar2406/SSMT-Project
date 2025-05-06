import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    MarianTokenizer,
    MarianMTModel,
    WhisperForConditionalGeneration
)
from model import CustomModel
import time

# ────────────────────────────────────────────────────────────────────────────────
# 1. Config / Paths
# ────────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_dir   = "weights/"
dataset_dir = os.path.join(cache_dir, "word-project")  # adjust if needed
checkpoint_path = "/share1/rohit.pawar/generation-models/ssmt/new_model/checkpoints_latest/checkpoint_epoch_MT53.pth"

whisper_name = "openai/whisper-medium"
mt_model_name = "Helsinki-NLP/opus-mt-en-hi"

# ────────────────────────────────────────────────────────────────────────────────
# 2. Instantiate processors, tokenizers, and models
# ────────────────────────────────────────────────────────────────────────────────
# 2a) Whisper processor + model (we only need the encoder in your CustomModel)
speech_processor = WhisperProcessor.from_pretrained(whisper_name, cache_dir=cache_dir)
whisper_model   = WhisperForConditionalGeneration.from_pretrained(whisper_name, cache_dir=cache_dir)

# 2b) Marian MT-model + tokenizer
marian_tokenizer = MarianTokenizer.from_pretrained(mt_model_name, cache_dir=cache_dir)
marian_model     = MarianMTModel.from_pretrained(mt_model_name,
                                                 add_cross_attention=False,
                                                 cache_dir=cache_dir)

# 2c) Your CustomModel
model = CustomModel(marian_model, whisper_model).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"✅ Loaded CustomModel from {checkpoint_path}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Inference function
# ────────────────────────────────────────────────────────────────────────────────
def infer_speech_to_hindi(
    model: torch.nn.Module,
    processor: WhisperProcessor,
    tokenizer: MarianTokenizer,
    audio_array: torch.Tensor, 
    english_text,          # shape: (n_samples,) or (1, n_samples)
    sampling_rate: int = 16000,
    max_len: int = 64,
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Run your CustomModel.generate on a single utterance and return the Hindi text.
    """
    # 3.1 Prepare Whisper features
    # Debugging point
    inputs = processor(audio_array,
                       sampling_rate=sampling_rate,
                       return_tensors="pt")
    speech_feats = inputs.input_features.to(device)       # (1, seq_len, feat_dim)
     
    # 3.2 Build dummy English encoder input (just a single PAD token so
    #     your cross-attention line still fires)
    # enc_ids   = torch.tensor([[tokenizer.pad_token_id]], device=device)
    # enc_mask  = torch.zeros_like(enc_ids, device=device)

    english_inputs = tokenizer(english_text, return_tensors="pt")

    enc_ids = english_inputs["input_ids"].squeeze(0).to(device)  # (seq_len)
    enc_mask = english_inputs["attention_mask"].squeeze(0).to(device)  # (seq_len)


    # 3.3 Prepare Hindi decoder start token
    bos_id    = int(61949)
    dec_ids   = torch.tensor([[bos_id]], device=device)
    dec_mask  = None

    #breakpoint()  # Debugging point
    # 3.4 Call your joint generate()
    with torch.no_grad():
        out_ids = model.generate(
            english_input_ids      = enc_ids.unsqueeze(0),
            hindi_input_ids        = dec_ids,
            speech_features        = speech_feats,
            hindi_attention_mask   = dec_mask,
            english_attention_mask = enc_mask.unsqueeze(0),
            max_new_tokens= 64,
            do_sample     = False,
            temperature   = 1.0,
            top_k         = 64
        )

    # 3.5 Decode to string
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]


# ────────────────────────────────────────────────────────────────────────────────
# 4. Run on your first example from disk
# ────────────────────────────────────────────────────────────────────────────────
raw_ds   = load_from_disk('weights/' + "word-poject")
# assume each item has a numpy array at ["chunked_audio_filepath"]["array"]

start = 9501
end = 10501

for i in range(start, end):
    start_time = time.time()
    waveform = raw_ds[i]["chunked_audio_filepath"]["array"]

    audio_tensor = waveform

    english_text = raw_ds[i]["text"]
    hindi_text   = raw_ds[i]["hi_text"]

    translation = infer_speech_to_hindi(
        model,
        speech_processor,
        marian_tokenizer,
        audio_tensor,
        english_text,
        sampling_rate=16000,
        max_len=128,
        device=device
    )
    # print("🗣️  Predicted Hindi translation:\n", translation)
    # print("Actully Hindi translation:\n", hindi_text)
    # print("English text:\n", english_text)

    with open("predicted_model_1000.txt", "a") as f:
        f.write(str(i) + "\n")
        f.write(translation+ "\n")

    with open("ground_model_1000.txt", "a") as f:
        f.write(str(i) + "\n")
        f.write(hindi_text + "\n")

    end_time = time.time()
    print(f"Time taken for {i} is {end_time - start_time} seconds")
    # with open("english_1000.txt", "a") as f:
    #     f.write(english_text + "\n")
