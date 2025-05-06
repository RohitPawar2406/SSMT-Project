from transformers import AutoTokenizer, MarianForCausalLM, MarianMTModel, MarianConfig, MarianTokenizer

mt_model_name = "Helsinki-NLP/opus-mt-en-hi"
cache_dir = "weights/"


# tokenizer = AutoTokenizer.from_pretrained(mt_model_name , cache_dir=cache_dir)
# # model = MarianForCausalLM.from_pretrained(mt_model_name, add_cross_attention=False, cache_dir=cache_dir)
# # assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
# # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# # outputs = model(**inputs)

# # logits = outputs.logits
# # expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
# # list(logits.shape) == expected_shape


model = MarianMTModel.from_pretrained(mt_model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(mt_model_name, cache_dir=cache_dir)

sample_text = "And the LORD spake unto Moses in the wilderness of Sinai, in the tabernacle of the congregation, on the first day of the second month, in the second year after they were come out of the land of Egypt."
batch = tokenizer([sample_text], return_tensors="pt")

batch = {k: v.to("cuda") for k, v in batch.items()}  # move inputs to CUDA
model = model.to("cuda")


generated_ids = model.generate(**batch)
g1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
with open("output.txt", "w") as f:
    f.write(g1)

# tokenizer = MarianTokenizer.from_pretrained(mt_model_name, cache_dir=cache_dir)
# src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
# tgt_texts = [" उपकरण आपके द्वारा चुनी गई भाषा में वेब पर कहीं भी लिखना आसान बनाता है"]  # optional

# inputs = tokenizer(src_texts, text_target=tgt_texts, return_tensors="pt", padding=True)

"""
(Pdb) inputs
{'input_ids': tensor([[   56,   489,    19,  1685, 17822,  1328,     3,     0],
        [21381,  1137,    62,  4796,    34,  3188,     3,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]]), 'labels': tensor([[2228,  522,  335, 7972,  485, 1152,   11, 2298,   33,  889,   47, 5208,
         2047, 4632,    5,    0]])}
"""

breakpoint()
 