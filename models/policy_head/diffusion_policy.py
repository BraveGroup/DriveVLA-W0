import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init


class PolicyHead(nn.Module):

    def __init__(self, action_dim, embedding_dim, width, noise_scheduler):
        """
        Initialize the PolicyHead.

        Args:
            action_dim (int): Dimensionality of the input actions.
            embedding_dim (int): Dimensionality of the encoded embeddings.
            width (int): Width of the intermediate layers.
            noise_scheduler (nn.Module): Flow matching scheduler to add noise.
        """
        super(PolicyHead, self).__init__()
        
        self.encoder = Emu3ActionEncoder(action_dim, embedding_dim, width)
        self.decoder = MultiLayerAttentionDecoder(embedding_dim, action_dim, num_heads=4, hidden_dim=128, num_layers=1)
        self.noise_scheduler = noise_scheduler

    def forward_loss(self, action_sequence, noise, tau):
        """
        Forward pass for action denoising.

        Args:
            action_sequence (torch.Tensor): Original actions of shape (batch_size, seq_len, action_dim).

        Returns:
            dict: Contains predicted actions and loss.
        """
        batch_size, seq_len, action_dim = action_sequence.shape

        # Add noise to the action sequence using the scheduler
        noisy_actions = self.noise_scheduler.add_noise(action_sequence, noise, tau)  # Both are (batch_size, seq_len, action_dim)

        # Encode the noisy actions
        embeddings = self.encoder(noisy_actions, tau)  # Shape: (batch_size, seq_len, embedding_dim)

        # Decode the embeddings back to action space
        velocity_pred = self.decoder(embeddings)  # Shape: (batch_size, seq_len, action_dim)

        # target
        target = action_sequence - noise

        # Compute the reconstruction loss (e.g., Mean Squared Error)
        loss = F.mse_loss(velocity_pred, target, reduction="mean")

        return {
            "loss": loss
        }
    
    def forward(self, z, t):
        """
        Forward pass for generating samples.

        Args:
            z (torch.FloatTensor): Initial latent variables.
            t (torch.FloatTensor): Timesteps for flow matching.

        Returns:
            torch.FloatTensor: Predicted noise for the given latent variables.
        """
        return self.decoder(self.encoder(z, t))
    
    def sample(self, z, noise_scheduler, num_steps, device, guidance_scale=1.0):
        """
        Generate samples using the flow matching process.

        Args:
            z (torch.FloatTensor): Initial latent variables.
            noise_scheduler (FlowMatchingScheduler): Noise scheduler instance.
            num_steps (int): Number of sampling steps.
            device (torch.device): Device for computation.
            guidance_scale (float): Scale for classifier-free guidance.

        Returns:
            torch.FloatTensor: Generated samples.
        """
        z = z.to(device)
        timesteps = torch.linspace(noise_scheduler.s, 0, num_steps, device=device)

        for i, t in enumerate(timesteps[:-1]):
            # Expand timestep for the current batch
            t_expanded = t.expand(z.shape[0],z.shape[1])

            # Predict noise
            pred_uncond = self.forward(z, t_expanded)  # Unconditional prediction
            pred_cond = pred_uncond  # Replace with conditional prediction if applicable
            
            if guidance_scale != 1.0:
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                v_pred = pred_uncond

            # Compute step size
            step_size = timesteps[i] - timesteps[i + 1]

            # Update latent variables
            z = z - step_size * v_pred

        return z

class RoPE(nn.Module):
    def __init__(self, embedding_dim):
        """
        Rotary Positional Encoding (RoPE).

        Args:
            embedding_dim (int): Dimensionality of the input embeddings.
        """
        super(RoPE, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, seq_len, device):
        """
        Generate rotary positional encoding.

        Args:
            seq_len (int): Length of the input sequence.
            device (torch.device): Device for tensor computation.

        Returns:
            torch.Tensor: Rotated positional encodings of shape (seq_len, embedding_dim).
        """
        # Create the position indices for the sequence length
        position_indices = torch.arange(0, seq_len, dtype=torch.float32, device=device)

        # Create the scaling factor for each dimension
        div_term = torch.exp(-torch.arange(0, self.embedding_dim, 2, dtype=torch.float32, device=device) * 
                             math.log(10000.0) / self.embedding_dim)

        # Apply scaling and rotation
        position_indices = position_indices.unsqueeze(-1) * div_term

        # Interleave sine and cosine for each position index
        encoding = torch.zeros(seq_len, self.embedding_dim, device=device)
        encoding[:, 0::2] = torch.sin(position_indices)  # Even indices: sine
        encoding[:, 1::2] = torch.cos(position_indices)  # Odd indices: cosine

        return encoding

class Emu3ActionEncoder(nn.Module):
    def __init__(self, action_dim, embedding_dim, width):
        """
        Initialize the Emu3ActionEncoder.

        Args:
            action_dim (int): Dimensionality of the input actions.
            embedding_dim (int): Output embedding dimensionality.
            width (int): Width of the MLP layers.
        """
        super(Emu3ActionEncoder, self).__init__()

        # Define the MLP layers
        self.W1 = nn.Linear(action_dim, width)
        self.W2 = nn.Linear(width + width, width)  # Concatenating 3 encodings
        self.W3 = nn.Linear(width, embedding_dim)

        # Define sinusoidal positional encoding for tau
        self.positional_encoding = PositionalEncoding(width)

        # Define 1D RoPE for sequence
        self.rope = RoPE(width)

    def forward(self, action_sequence, tau):
        """
        Forward pass for encoding actions.

        Args:
            action_sequence (torch.Tensor): Input actions of shape (batch_size, seq_len, action_dim).
            tau (torch.Tensor): Flow matching timesteps of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Encoded embeddings of shape (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len, action_dim = action_sequence.shape

        # Apply the first linear layer (W1) to project action_dim to width
        action_projection = self.W1(action_sequence)  # Shape: (batch_size, seq_len, width)

        # Expand tau to match the sequence length
        tau = tau.expand(-1, seq_len)  # Shape: (batch_size, seq_len)

        # Get positional encoding for tau
        tau_encoding = self.positional_encoding(tau)  # Shape: (batch_size, seq_len, width)
        
        # Get 1D RoPE encoding for the sequence
        rope_encoding = self.rope(seq_len, action_sequence.device)  # Shape: (seq_len, width)

        # Expand the RoPE encoding to match the batch size
        rope_encoding = rope_encoding.unsqueeze(0).expand(batch_size, seq_len, self.rope.embedding_dim)

        action_projection = action_projection + rope_encoding

        # Concatenate action projection, tau encoding, and RoPE encoding
        concat_features = torch.cat([action_projection, tau_encoding], dim=-1)  # Shape: (batch_size, seq_len, 3 * width)

        # Apply the second linear layer (W2) and activation (Swish)
        transformed_features = F.silu(self.W2(concat_features))  # Shape: (batch_size, seq_len, width)

        # Apply the third linear layer (W3) to get final embeddings
        embeddings = self.W3(transformed_features)  # Shape: (batch_size, seq_len, embedding_dim)

        return embeddings



class PositionalEncoding(nn.Module):
    def __init__(self, width):
        """
        Sinusoidal positional encoding function.

        Args:
            width (int): Dimensionality of the positional encoding. Must be even.
        """
        super(PositionalEncoding, self).__init__()
        assert width % 2 == 0, "Width must be even for sinusoidal positional encoding."
        self.width = width

    def forward(self, tau):
        """
        Generate positional encoding for input timesteps.

        Args:
            tau (torch.Tensor): Input timesteps of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Positional encodings of shape (batch_size, seq_len, width).
        """
        batch_size, seq_len = tau.shape
        device = tau.device

        # Create position indices (evenly spaced dimensions)
        position_indices = torch.arange(0, self.width, 2, dtype=torch.float32, device=device)  # Half the width
        div_term = torch.exp(-position_indices * math.log(10000.0) / self.width)

        # Apply tau scaling for sinusoidal encoding
        tau = tau.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
        sinusoidal = tau * div_term  # Shape: (batch_size, seq_len, width // 2)

        # Compute sine and cosine values interleaved
        encoding = torch.zeros(batch_size, seq_len, self.width, device=device)
        encoding[..., 0::2] = torch.sin(sinusoidal)  # Even indices: sine
        encoding[..., 1::2] = torch.cos(sinusoidal)  # Odd indices: cosine

        return encoding

class MultiLayerAttentionDecoder(nn.Module):
    def __init__(self, embedding_dim, action_dim, num_heads, hidden_dim, num_layers=1):
        """
        Multi-layer attention decoder with layer normalization and MLP.

        Args:
            embedding_dim (int): Input embedding dimension.
            action_dim (int): Output action dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): MLP hidden layer dimension.
            num_layers (int): Number of stacked attention layers.
        """
        super(MultiLayerAttentionDecoder, self).__init__()

        self.num_layers = num_layers

        # Attention layers and layer norms
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])

        # MLP to map attention output to actions
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self._initialize_parameters()

    def forward(self, embeddings):
        """Forward pass through attention layers and MLP."""
        for i in range(self.num_layers):
            attn_output, _ = self.attention_layers[i](embeddings, embeddings, embeddings)
            embeddings = self.layer_norms[i](attn_output)  # Layer norm after attention

        return self.mlp(embeddings)

    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Xavier initialization for attention layers
        for i in range(self.num_layers):
            init.xavier_uniform_(self.attention_layers[i].in_proj_weight)
            init.xavier_uniform_(self.attention_layers[i].out_proj.weight)
            init.zeros_(self.attention_layers[i].in_proj_bias)
            init.zeros_(self.attention_layers[i].out_proj.bias)

        # Xavier initialization for MLP layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

        # Initialize layer norm weights and biases
        for norm in self.layer_norms:
            init.ones_(norm.weight)
            init.zeros_(norm.bias)
