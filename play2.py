from transformers import MarianMTModel

mt_model_name = "Helsinki-NLP/opus-mt-en-hi"
cache_dir = "weights/"
# assume you have already loaded your model:
model = MarianMTModel.from_pretrained(mt_model_name, cache_dir=cache_dir)

# 1. Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# 2. Unfreeze the last two decoder layers entirely
decoder = model.get_decoder()  # huggingface MarianMTModel stores the decoder here


for layer in decoder.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True


# 3. (Redundant, but explicit) Ensure the four submodules are trainable
# for layer in decoder.layers[4:-1]:
#     for sub in [layer.self_attn_layer_norm,
#                 layer.fc1,
#                 layer.fc2,
#                 layer.final_layer_norm]:
#         for param in sub.parameters():
#             param.requires_grad = True

# Now verify
for i, layer in enumerate(decoder.layers):
    trainable = any(p.requires_grad for p in layer.parameters())
    print(f"Layer {i}: {'trainable' if trainable else 'frozen'}")


breakpoint()
