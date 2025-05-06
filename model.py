from transformers import MarianMTModel, WhisperForConditionalGeneration, MarianTokenizer
from torch import nn
import torch
import torch.nn.functional as F

cache_dir   = "weights/"
mt_model_name = "Helsinki-NLP/opus-mt-en-hi"
marian_tokenizer = MarianTokenizer.from_pretrained(mt_model_name, cache_dir=cache_dir)

class CustomModel(nn.Module):
    def __init__(self, marian_model, Whisper_model):
        super(CustomModel, self).__init__()
        self.marian_encoder = marian_model.get_encoder() # output shape: (batch_size, sequence_length, 512)
        self.marian_decoder = marian_model.get_decoder() # output shape: (batch_size, sequence_length, 512)

        self.whisper_encoder = Whisper_model.get_encoder() # output shape: (batch_size, 1500, 1024)

        self.cross_alignment_linear = nn.Linear(1024, 512)  # convert speech features to match Marian's hidden size from 1024 to 512
        
        #self.layer_fn = nn.LayerNorm(marian_model.config.d_model, eps=1e-6)
        #self.output_layer = nn.Linear(marian_model.config.d_model, marian_model.config.vocab_size)  # (hidden_dim → vocab_size)

        self.output_layer = marian_model.lm_head

        for param in self.output_layer.parameters():
            param.requires_grad = True
            
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        for param in self.marian_encoder.parameters():
            param.requires_grad = False

        for param in self.marian_decoder.parameters():
            param.requires_grad = False

        for layer in self.marian_decoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True


    def forward(self, english_input_ids, hindi_input_ids, speech_features, hindi_attention_mask=None, english_attention_mask=None):
        """
            english_input_ids: (batch_size, 128)
            speech_features: (batch_size, 80, 3000)
            hindi_input_ids: (batch_size, 128)
            hindi_attention_mask: (batch_size, 128)
            english_attention_mask: (batch_size, 128)
        """
        # Encoder step
        if speech_features is not None:
            whisper_encoder_outputs = self.whisper_encoder(speech_features).last_hidden_state # (B, 1500, 1024)
        marian_encoder_outputs = self.marian_encoder(english_input_ids, attention_mask=english_attention_mask).last_hidden_state # (B, 128, 512)

        

        if speech_features == None:
            combined_encoder_outputs = marian_encoder_outputs
        else:
            # Cross alignment
            cross_attention_outs = self.cross_alignment_linear(whisper_encoder_outputs)  # (B, 1500, 512)
            # Concatenate the outputs from the Whisper encoder and Marian encoder
            combined_encoder_outputs = torch.cat((marian_encoder_outputs, cross_attention_outs), dim=1)  # (B, 1500 + 128, 512)   

        # Pass the combined outputs to the decoder
        marian_decoder_outputs = self.marian_decoder(hindi_input_ids, encoder_hidden_states=combined_encoder_outputs, attention_mask=hindi_attention_mask).last_hidden_state # (B, 128, 512)

        logits = self.output_layer(marian_decoder_outputs)  # (B, 128, vocab_size) # vocab_size = 61950

        if self.training:
            return logits
        else:  
            return logits
        
    def generate(self, english_input_ids, hindi_input_ids, speech_features, hindi_attention_mask, english_attention_mask, max_new_tokens=25, temperature=0.9, do_sample=True, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        english_input_ids, hindi_input_ids, speech_features, hindi_attention_mask=None, english_attention_mask=None
        """
        idx = hindi_input_ids
        for j111 in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            logits = self(english_input_ids, idx_cond, speech_features, hindi_attention_mask, english_attention_mask)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next.item == "</s>":
                break 

            #print(f"{j111: }: ", marian_tokenizer.decode(idx_next.item()))

        return idx 

        

if __name__ == "__main__":

    mt_model_name = "Helsinki-NLP/opus-mt-en-hi"
    cache_dir = "weights/"
    model_medium =  "openai/whisper-medium" 
    cache_dir_whisper = "weights"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Marian_model = MarianMTModel.from_pretrained(mt_model_name, add_cross_attention=False, cache_dir=cache_dir)
    Whisper_model = WhisperForConditionalGeneration.from_pretrained(model_medium, cache_dir=cache_dir)

    english_src_inputs = torch.randint(0, 1000, (2, 128), device=device)  # (batch_size, sequence_length)
    speech_features = torch.randn(2, 80, 3000).to(device=device)  # (batch_size, 80, 3000)
    hindi_src_ids = torch.randint(0, 1000, (2, 128), device=device)  # (batch_size, sequence_length)
    hindi_target_ids = torch.randint(0, 1000, (2, 128), device=device)  # (batch_size, sequence_length)
    attention_mask = torch.ones(2, 128).to(device=device)  # (batch_size, sequence_length)

    model = CustomModel(Marian_model, Whisper_model)
    model = model.to("cuda")

    model(english_src_inputs, hindi_src_ids, speech_features, attention_mask)
    breakpoint()
 