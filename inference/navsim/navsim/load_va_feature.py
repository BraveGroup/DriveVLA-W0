import os
import torch
from torchvision.utils import save_image
import pickle
import matplotlib.pyplot as plt
import numpy as np

from tokenizer.tokenizer_image.vq_model import VQ_models

# =============== VQ Model Settings =================== #
vq_model_name = "VQ-16"
codebook_size = 16384
codebook_embed_dim = 8
vq_ckpt = "pretrained_models/vq_ds16_t2i.pt"
latent_height = 18
latent_width = 32
qzshape = [12, codebook_embed_dim, latent_height, latent_width]  # [total_frame, args.codebook_embed_dim, latent_height, latent_width]


# ============== VQ Model Loading ================ #
vq_model = VQ_models[vq_model_name](
    codebook_size=codebook_size,
    codebook_embed_dim=codebook_embed_dim)
vq_model.eval()
checkpoint = torch.load(vq_ckpt, map_location="cpu")
vq_model.load_state_dict(checkpoint["model"])
del checkpoint
print("image tokenizer is loaded")


# =================== Vision Tokens Loading ============ #
tokens_path = "/home/hongxiao.yu/LlamaVideo_Official/data/navtrain_va_tokens/results_0060000.pt/sample_i2i_(12+0x11)x288x512xVQ-16"

pkls = os.listdir(path=tokens_path)
print(f"Found training tokens: {len(pkls)}")

# load one va token file
with open(os.path.join(tokens_path, pkls[17890]), "rb") as f:
    # vision tokens: (12, 18, 32)
    va_tokens = pickle.load(f)

# ================== Get Vision Codebook Feature ============== #
quant_b = vq_model.quantize.get_codebook_entry(va_tokens["vision"].cpu(), qzshape, channel_first=True)


# ================== Decode to Images ==================== #
sample = vq_model.decode_code(va_tokens["vision"].cpu(), qzshape)  # (12, 3, 288, 512)
save_image(sample, "samples.png", nrow=4, normalize=True, value_range=(-1, 1))



# =================== Action Tokens Setting ============ #
num_action_bins = 128
ACTION_MIN = torch.tensor([-0.0425, -0.3351, -0.2401], dtype=torch.float32)
ACTION_MAX = torch.tensor([7.0145, 0.3351, 0.2401], dtype=torch.float32)

# =================== Action Tokens Loading ============ #
sampled_action_tokens = va_tokens["action"].cpu()


# =================== Action Tokens Converting ============ #
def convert_tokenized_action_to_float_action(
    tokenized_action, action_min, action_max, vocab_size, num_action_bins):
    assert isinstance(tokenized_action, torch.Tensor)
    
    # Convert back from token indices to float values
    float_action = (tokenized_action - vocab_size).float() / (num_action_bins - 1) * (action_max - action_min) + action_min
    return float_action

sampled_action_tokens[:, 1] -= num_action_bins  # Adjust back for multiple action components
sampled_action_tokens[:, 2] -= 2 * num_action_bins  # Adjust back for multiple action components
saved_action_tokens = sampled_action_tokens
converted_action_tokens = convert_tokenized_action_to_float_action(sampled_action_tokens, ACTION_MIN, ACTION_MAX, codebook_size, num_action_bins)
plt.figure()
plt.plot(converted_action_tokens[:, 0], converted_action_tokens[:, 1])
plt.scatter(converted_action_tokens[:, 0], converted_action_tokens[:, 1])
plt.axis('equal')
plt.savefig("actions.png")
print("predicted action: ", converted_action_tokens)  # (8, 3)