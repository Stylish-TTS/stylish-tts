import math
from typing import List, Optional, Tuple
import torch
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange


class SpecDiscriminator(torch.nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm
        self.discriminators = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                    )
                ),
            ]
        )
        self.relu = torch.nn.LeakyReLU(0.1)

        self.out = norm_f(torch.nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = self.relu(y)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


def run_discriminator_model(disc, target, pred):
    y_d_rs = []
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []

    y_d_r, fmap_r = disc(target)
    y_d_g, fmap_g = disc(pred)
    y_d_rs.append(y_d_r)
    fmap_rs.append(fmap_r)
    y_d_gs.append(y_d_g)
    fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


#################################################


class WindowwiseTransformer(torch.nn.Module):
    """Process entire window using transformer architecture with sinusoidal position encodings.
    Allows cross-frame attention within the window while maintaining position-awareness.
    """

    def __init__(
        self,
        input_dim,
        context_dim,
        frames_per_window,
        num_context_layers=4,
        context_dropout=0.1,
        num_transformer_heads=8,
    ):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_dim, context_dim)

        # Generate fixed sinusoidal position encodings
        self.register_buffer(
            "pos_encoding",
            self._create_sinusoidal_encoding(frames_per_window, context_dim),
        )

        self.dropout = torch.nn.Dropout(context_dropout)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerEncoderLayer(
                    d_model=context_dim,
                    nhead=num_transformer_heads,
                    dropout=context_dropout,
                    batch_first=True,
                    dim_feedforward=context_dim * 4,
                    norm_first=True,  # Pre-norm architecture for better stability
                )
                for _ in range(num_context_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(context_dim)

        # Initialize a scale factor for position encodings
        self.pos_encoding_scale = torch.nn.Parameter(torch.ones(1))

    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal position encodings.

        Args:
            max_len: Maximum sequence length (frames_per_window)
            d_model: Embedding dimension (final_projection_dim)

        Returns:
            pos_encoding: Positional encoding matrix of shape (1, max_len, d_model)
        """
        pe = torch.zeros(int(max_len), d_model)
        position = torch.arange(0, int(max_len), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and normalize
        pe = pe.unsqueeze(0)

        return pe

    def _get_pos_encoding_subset(self, seq_len):
        """Get position encodings for the actual sequence length."""
        return self.pos_encoding[:, :seq_len, :]

    def forward(self, x):
        """
        Forward pass with scaled positional encodings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, last_cnn_output_dim)

        Returns:
            output: Processed tensor of shape (batch_size, seq_len, final_projection_dim)
        """
        x = self.input_projection(x)

        # Get positional encodings for the actual sequence length
        pos_enc = self._get_pos_encoding_subset(x.size(1))

        # Add scaled positional encodings
        x = x + (self.pos_encoding_scale * pos_enc)

        # Apply dropout after position encoding
        x = self.dropout(x)

        fmap = []
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
            fmap.append(x)

        return self.norm(x), fmap

    def reset_parameters(self):
        """Reset learnable parameters while keeping position encodings fixed."""
        torch.nn.init.normal_(self.pos_encoding_scale, mean=1.0, std=0.1)

        # Reset input projection
        torch.nn.init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            torch.nn.init.zeros_(self.input_projection.bias)

        # Reset transformer layers
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)


class ContextFreeEmbedding(torch.nn.Module):
    def __init__(
        self,
        input_wav_length=None,
        CNN_n_channels=None,
        CNN_dropout_rate=None,
        window_layers_dim=None,
        window_layers_num=None,
        window_layers_heads=None,
        window_dropout=None,
        noise_level=None,
    ):
        """
        Initialize the empty model
        """
        super().__init__()

        if input_wav_length is not None:
            self.config_model(
                input_wav_length,
                CNN_n_channels,
                CNN_dropout_rate,
                window_layers_dim,
                window_layers_num,
                window_layers_heads,
                window_dropout,
                noise_level,
            )
            self.make_model()

    def config_model(
        self,
        input_wav_length,
        CNN_n_channels,
        CNN_dropout_rate,
        window_layers_dim,
        window_layers_num,
        window_layers_heads,
        window_dropout,
        noise_level,
    ):
        """
        requires hp.CNN_n_channels, hp.CNN_dropout_rate, hp.window_layers_dim, hp.window_layers_num, hp.window_layers_heads, hp.window_dropout, hp.noise_level, hp.phoneme_classesm, input_wav_length
        """

        self.config = {
            "n_channels": CNN_n_channels,
            "dropout_rate": CNN_dropout_rate,
            "window_layers_dim": window_layers_dim,
            "window_layers_num": window_layers_num,
            "window_layers_heads": window_layers_heads,
            "window_dropout": window_dropout,
            "noise_level": noise_level,
            "input_wav_length": input_wav_length,
        }

    def load_config_state_dict(self, config_dict):
        self.config = config_dict
        # self.make_model()

    def save_config_state_dict(self):
        # print("Saving model with input_wav_length:", self.input_wav_length)
        return self.config

    def make_model(self):

        # configurable dims:
        bias = False
        n_channels = self.config["n_channels"]
        cnn_dropout_rate = self.config["dropout_rate"]

        window_layers_dim = self.config["window_layers_dim"]
        window_layers_num = self.config["window_layers_num"]
        window_layers_heads = self.config["window_layers_heads"]
        window_dropout = self.config["window_dropout"]

        noise_level = self.config["noise_level"]

        # calculated dims
        last_cnn_output_dim = n_channels * 4

        self.noise_level = noise_level
        self.input_wav_length = int(self.config["input_wav_length"])
        # Sanity checks

        assert self.input_wav_length > (0.005 * 16000)
        assert window_layers_dim <= last_cnn_output_dim

        # Feature Extractor - Fine-tuned for 8-10 frames per window, 20ms temporal resolution
        self.feature_extractor = torch.nn.Sequential(
            # Layer 1: Stride of 4
            torch.nn.Conv1d(
                1, n_channels, kernel_size=11, stride=4, padding=5, bias=bias
            ),
            torch.nn.BatchNorm1d(n_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(cnn_dropout_rate),
            # Layer 2: Stride of 4
            torch.nn.Conv1d(
                n_channels,
                n_channels * 2,
                kernel_size=11,
                stride=4,
                padding=5,
                bias=bias,
            ),
            torch.nn.BatchNorm1d(n_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(cnn_dropout_rate),
            # Layer 3: Stride of 2
            torch.nn.Conv1d(
                n_channels * 2,
                n_channels * 4,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias,
            ),
            torch.nn.BatchNorm1d(n_channels * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(cnn_dropout_rate),
            # Layer 4: Stride of 2
            torch.nn.Conv1d(
                n_channels * 4,
                last_cnn_output_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias,
            ),
            torch.nn.BatchNorm1d(last_cnn_output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(cnn_dropout_rate),
        )

        # Frequency attention mechanism (unchanged)
        self.freq_attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Conv1d(last_cnn_output_dim, last_cnn_output_dim, 1, bias=True),
            torch.nn.Sigmoid(),
        )

        # Temporal stream - Modified for 20ms temporal field
        self.temporal_stream = torch.nn.Sequential(
            # Broad temporal context (reduced kernel size due to halved temporal resolution)
            torch.nn.Conv1d(
                last_cnn_output_dim,
                last_cnn_output_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=8,
                bias=True,
            ),
            torch.nn.BatchNorm1d(last_cnn_output_dim),
            torch.nn.GELU(),
            # Fine detail processing (reduced kernel size due to halved temporal resolution)
            torch.nn.Conv1d(
                last_cnn_output_dim,
                last_cnn_output_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=8,
                bias=True,
            ),
        )

        # Spectral stream (unchanged)
        self.spectral_stream = torch.nn.Sequential(
            torch.nn.Conv1d(
                last_cnn_output_dim,
                n_channels * 12,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=8,
                bias=True,
            ),
            torch.nn.BatchNorm1d(n_channels * 12),
            torch.nn.GELU(),
            torch.nn.Conv1d(
                n_channels * 12,
                last_cnn_output_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=8,
                bias=True,
            ),
            torch.nn.BatchNorm1d(last_cnn_output_dim),
            torch.nn.GELU(),
        )

        # Feature fusion (unchanged)
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv1d(last_cnn_output_dim * 2, last_cnn_output_dim, 1, bias=True),
            torch.nn.BatchNorm1d(last_cnn_output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
        )

        assert last_cnn_output_dim == window_layers_dim * 2
        self.frames_per_window = 16
        # self.layer_dims = self.make_layer_sizer()
        # self.layer_dims = model_utils.ModelUtils.extract_layer_dims(self)
        # self.frames_per_window = model_utils.ModelUtils.calculate_layer_sizes(self.layer_dims, torch.tensor([self.input_wav_length]), -1)[0].int()
        # self.model_utils = model_utils.ModelUtils(self.layer_dims, self.input_wav_length, self.frames_per_window)

        # Window processor (unchanged)
        self.window_processor = WindowwiseTransformer(
            input_dim=last_cnn_output_dim,
            context_dim=window_layers_dim,
            frames_per_window=self.frames_per_window,
            num_context_layers=window_layers_num,
            context_dropout=window_dropout,
            num_transformer_heads=window_layers_heads,
        )

        # Final classifier (added Relu and dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(window_layers_dim, window_layers_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(window_layers_dim * 4, 1),
        )

    # def update_frames_per_window(self, input_wav_length):
    #     self.input_wav_length = int(input_wav_length)
    #     self.config['input_wav_length'] = self.input_wav_length
    #     self.frames_per_window = self.model_utils.calculate_layer_sizes(self.layer_dims, torch.tensor([self.input_wav_length]), -1)[0].int()
    #     self.frames_per_window = torch.ceil((self.frames_per_window)).int()
    #     #print("frames_per_window (frames per clip if disable_windowing):", self.frames_per_window.item())
    #     return self.frames_per_window

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * self.noise_level

        x = x.unsqueeze(1)  # Add channel dim

        # Feature extraction (B, 1, T) -> (B, 8n, T')
        features = self.feature_extractor(x)

        # Attention
        att = self.freq_attention(features)
        features = features * att

        # Dual stream processing
        temporal = self.temporal_stream(features)
        spectral = self.spectral_stream(features)

        # Combine streams and fuse
        fused = torch.cat([temporal, spectral], dim=1)
        fused = self.fusion(fused)

        # Prepare for transformer
        fused = fused.transpose(1, 2)  # (B, T', 8n)

        # Apply window processor
        features, fmap = self.window_processor(fused)

        # Classify each frame
        logits = self.classifier(features)

        return logits, fmap


def slice_windows(
    audio_batch: torch.Tensor,
    *,
    window_size: int,
    stride: int,
    # sample_rate: int = 16000,
    # window_size_ms: int = 160,
    # stride_ms: int = 80
) -> torch.Tensor:
    """
    Create fixed-size windows with overlap from a batch of audio sequences using vectorized operations.

    Args:
        audio_batch: Input audio of shape [batch_size, 1, max_audio_length]
        sample_rate: Audio sample rate in Hz
        window_size_ms: Window size in milliseconds
        stride_ms: Stride size in milliseconds

    Returns:
        Tensor of shape [batch_size, num_windows, window_size]
    """
    # audio_batch = audio_batch.squeeze(1)  # [batch_size, max_audio_length]
    batch_size, max_audio_length = audio_batch.shape

    # Calculate window parameters
    # window_size = int(window_size_ms * sample_rate / 1000)
    # stride = int(stride_ms * sample_rate / 1000)
    num_windows = ((max_audio_length - window_size) // stride) + 1

    # Create indices for all windows at once
    offsets = torch.arange(0, window_size, device=audio_batch.device)
    starts = torch.arange(0, num_windows * stride, stride, device=audio_batch.device)

    # Create a indices matrix [num_windows, window_size]
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

    # Handle out-of-bounds indices
    valid_indices = indices < max_audio_length
    indices = torch.minimum(
        indices, torch.tensor(max_audio_length - 1, device=audio_batch.device)
    )

    # Expand indices for batching [batch_size, num_windows, window_size]
    batch_indices = torch.arange(batch_size, device=audio_batch.device)[:, None, None]

    # Gather windows using expanded indices
    windows = audio_batch[batch_indices, indices]

    # Zero out invalid regions
    windows = windows * valid_indices.float()

    return windows


class ContextFreeDiscriminator(torch.nn.Module):

    def __init__(
        self,
    ):
        super(ContextFreeDiscriminator, self).__init__()
        self.embedding = ContextFreeEmbedding(
            input_wav_length=1024,
            CNN_n_channels=64,
            CNN_dropout_rate=0.1,
            window_layers_dim=128,
            window_layers_num=4,
            window_layers_heads=8,
            window_dropout=0.25,
            noise_level=0.0,
        )

    def forward(self, x):
        x = slice_windows(x, window_size=1024, stride=512)
        time_steps = x.shape[1]
        x = rearrange(x, "b t w -> (b t) w")
        x, fmap = self.embedding(x)
        x = rearrange(x, "(b t) f c -> b (t f c)", t=time_steps)
        return x, []  # fmap
