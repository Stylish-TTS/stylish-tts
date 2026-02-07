import math
import torch.nn as nn
import torch
from typing import Callable, Optional, Union, Unpack
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    Qwen3Config,
)
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers import Cache
from kanade_tokenizer.model import GlobalEncoder


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time):
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Qwen3AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.to_weight = nn.Linear(hidden_size, hidden_size)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        text_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.to_weight(cond).unsqueeze(1) * hidden_states.to(text_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3NAREncoderLayer(torch.nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.self_attn.is_causal = False  # Encoder is non-causal

        self.mlp = Qwen3MLP(config)
        self.text_layernorm = Qwen3AdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3AdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.text_layernorm(hidden_states, cond)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3NARModel(torch.nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        config._attn_implementation = "sdpa"
        self.layers = nn.ModuleList(
            [
                Qwen3NAREncoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3AdaptiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(self, texts_embeds, cond, attention_mask=None):
        hidden_states = texts_embeds
        position_ids = torch.arange(
            0, hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                cond,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states, cond)
        return hidden_states


class MaskedTokenPredictor(nn.Module):
    def __init__(
        self, text_vocab, output_vocab, hidden_layers=8, attn_heads=1 * 4, kv_heads=1
    ):
        super().__init__()
        self.text_vocab = text_vocab + 1
        self.output_vocab = output_vocab + 1
        self.text_mask_token = text_vocab
        self.output_mask_token = output_vocab
        # self.speaker_mask_id = 355
        self.ref_mask = 0

        self.embed_text = nn.Embedding(self.text_vocab, 256)
        self.embed_output = nn.Embedding(self.output_vocab, 256)
        self.t_embed = TimestepEmbedder(512)
        self.input_proj = nn.Linear(256 * 2 + 128, 512)
        self.model = Qwen3NARModel(
            Qwen3Config(
                hidden_size=512,
                intermediate_size=512 * 3,
                num_hidden_layers=hidden_layers,
                num_attention_heads=attn_heads,
                num_key_value_heads=kv_heads,
                head_dim=128,
                max_window_layers=0,
                use_cache=False,
                use_sliding_window=False,
                max_position_embeddings=2048,
            )
        )
        self.lm_head = nn.Linear(512, self.output_vocab)

    def mask_prob(self, t):
        return torch.cos(t * torch.pi / 2)

    def forward(self, text, ref, output, t, output_hidden_state=False):
        text = self.embed_text(text)
        style = ref.unsqueeze(1).repeat(1, text.shape[1], 1)
        # style = self.embed_speaker(ref).unsqueeze(1).repeat(1, text.shape[1], 1)
        output = self.embed_output(output)
        t = self.t_embed(t)
        x = self.input_proj(torch.cat([text, style, output], -1))
        x = self.model(x, t)
        if output_hidden_state:
            return x
        else:
            return self.lm_head(x)

    def make_noisy_sample(self, text, ref, output, cond_dropout=0.1):
        B, T = output.shape
        device = output.device

        drop_mask = torch.rand(B, 1, device=device) < cond_dropout
        # prompt_len = torch.randint(min(T//4, 12), T//2, (1,)).item()
        # noisy_ref = ref[:, :prompt_len, :].masked_fill(drop_mask.unsqueeze(-1), 0)
        noisy_ref = ref.masked_fill(drop_mask, self.ref_mask)
        noisy_text = text.masked_fill(drop_mask, self.text_mask_token)

        t = torch.rand(B, 1, device=device)
        mask_prob = self.mask_prob(t)
        corrupt_mask = torch.bernoulli(mask_prob.expand(B, T)).bool()
        noisy_output = output.masked_fill(corrupt_mask, self.output_mask_token)
        return noisy_text, noisy_ref, noisy_output, corrupt_mask, t.squeeze(1)

    @torch.no_grad()
    def generate(
        self,
        text,
        ref,
        output,
        T_prompt=None,
        steps=8,
        cfg_scale=0,
        temp=1,
        gumbel_sampling=False,
    ):
        # Helper: Gumbel Noise for "Confidence Randomization"
        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -torch.log(-torch.log(noise + 1e-10) + 1e-10)

        B, T_total = output.shape
        if T_prompt is None:
            T_prompt, T_tgt = 0, T_total
        else:
            T_tgt = T_total - T_prompt
        for step in range(steps):
            t = torch.tensor([step / steps], device=text.device)
            if cfg_scale > 0:
                h_cond = self(text, ref, output, t.repeat(B), output_hidden_state=True)[
                    :, T_prompt:
                ]
                h_uncond = self(
                    torch.full_like(text[:, T_prompt:], self.text_mask_token),
                    torch.full_like(ref, self.ref_mask),
                    output[:, T_prompt:],
                    t.repeat(B),
                    output_hidden_state=True,
                )
                h_cfg = h_cond + cfg_scale * (h_cond - h_uncond)
                logits = self.lm_head(h_cfg)
            else:
                logits = self(text, ref, output, t.repeat(B), output_hidden_state=False)
                logits = logits[:, T_prompt:, :]

            if gumbel_sampling:
                current_temp = max(temp * (1 - t), 0.001)
                logits = logits + gumbel_noise(logits) * current_temp
                candidate_ids = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                candidate_probs = torch.gather(
                    probs, -1, candidate_ids.unsqueeze(-1)
                ).squeeze(-1)
                candidate_probs = (
                    candidate_probs + gumbel_noise(candidate_probs) * current_temp
                )
            else:
                probs = nn.functional.softmax(logits / temp, dim=-1)
                candidate_ids = torch.argmax(probs, dim=-1)  # [B, T_tgt]
                candidate_probs = torch.gather(
                    probs, -1, candidate_ids.unsqueeze(-1)
                ).squeeze(-1)
            ratio_mask = self.mask_prob(t + 1 / steps).item()
            n_tokens_to_mask = int(T_tgt * ratio_mask)

            if n_tokens_to_mask > 0:
                n_keep = T_tgt - n_tokens_to_mask
                _, keep_indices = torch.topk(candidate_probs, n_keep, dim=-1)
                mask_next = torch.ones_like(candidate_ids, dtype=torch.bool)
                mask_next.scatter_(1, keep_indices, False)
                new_target_part = candidate_ids.clone()
                new_target_part[mask_next] = self.output_mask_token
                output[:, T_prompt:] = new_target_part
            else:
                output[:, T_prompt:] = candidate_ids
        return output[:, T_prompt:]


# class TokenPredictor(nn.Module):
#     def __init__(self, inter_dim, style_dim):
#         super().__init__()
#         self.acoustic_vocab_size = 12800 + 1
#         self.semantic_vocab_size = 178
#         self.mask_token = 12800
#         self.num_speaker = 355

#         self.embed_semantic = nn.Embedding(self.acoustic_vocab_size, 256)
#         # self.embed_semantic = nn.Embedding(32, inter_dim)
#         self.embed_text = nn.Embedding(self.semantic_vocab_size, 128)
#         self.embed_speaker = nn.Embedding(self.num_speaker, 128)
#         self.t_embed = TimestepEmbedder(512)
#         self.input_proj = nn.Linear(512, 512)
#         self.model = Qwen3NARModel(config)
#         self.lm_head = nn.Linear(512, self.acoustic_vocab_size)
#         self.prompt_dropout = 0.2

#     def forward(self, acoustic, semantic, spk_id, t, alignment, output_hidden_state=False):
#         acoustic = self.embed_semantic(acoustic)
#         semantic = (self.embed_text(semantic).mT @ alignment).mT
#         t = self.t_embed(t)
#         spk = self.embed_speaker(spk_id).unsqueeze(1).repeat(1, acoustic.shape[1], 1)
#         x = self.input_proj(torch.cat([acoustic, semantic, spk], -1))
#         x = self.model(x, t)
#         return x if output_hidden_state else self.lm_head(x)

#     def mask_prob(self, t):
#         return torch.cos(t * torch.pi / 2)

#     def make_noisy_sample(self, gt_codes):
#         B, T = gt_codes.shape
#         t = torch.rand(B, 1, device=gt_codes.device)
#         corrupt_mask = torch.bernoulli(self.mask_prob(t).expand(B, T)).bool()
#         noisy_codes = gt_codes.masked_fill(corrupt_mask, self.mask_token)
#         return noisy_codes, corrupt_mask, t.squeeze(1)

#     @torch.no_grad()
#     def generate(
#         self,
#         prompt_semantic,
#         prompt_acoustic,
#         semantic,
#         style,
#         alignment,
#         steps=8,
#         cfg_scale=0,
#         temp=1,
#         gumbel_sampling=False,
#     ):
#         B, T_tgt = semantic.shape
#         T_tgt = alignment.shape[-1]
#         # T_prompt = prompt_acoustic.shape[-1]

#         # Helper: Gumbel Noise for "Confidence Randomization"
#         def gumbel_noise(t):
#             noise = torch.zeros_like(t).uniform_(0, 1)
#             return -torch.log(-torch.log(noise + 1e-10) + 1e-10)

#         # semantic = torch.cat([prompt_semantic, semantic], 1)
#         acoustic = torch.full(
#             (B, T_tgt), self.mask_token, dtype=torch.long, device=semantic.device
#         )
#         # acoustic = torch.cat([prompt_acoustic, acoustic], 1)

#         for step in range(steps):
#             t = torch.tensor([step / steps], device=semantic.device)
#             if cfg_scale > 0:
#                 h_cond = self(
#                     acoustic, semantic, style, t.repeat(B), alignment, output_hidden_state=True
#                 )[:, T_prompt:]
#                 h_uncond = self(
#                     acoustic[:, T_prompt:],
#                     semantic[:, T_prompt:, :],
#                     None,
#                     t.repeat(B),
#                     alignment,
#                     output_hidden_state=True,
#                 )
#                 h_cfg = h_cond + cfg_scale * (h_cond - h_uncond)
#                 # h_rescaled = h_cfg * (h_cond.std() / (h_cfg.std() + 1e-6))
#                 # h_cfg = rescale_cfg * h_rescaled + (1 - rescale_cfg) * h_cfg
#                 logits = self.lm_head(h_cfg)
#             else:
#                 logits = self(
#                     acoustic, semantic, style, t.repeat(B), alignment, output_hidden_state=False
#                 )
#                 logits = logits# [:, T_prompt:, :]

#             if gumbel_sampling:
#                 current_temp = max(temp * (1 - t), 0.001)
#                 logits = logits + gumbel_noise(logits) * current_temp
#                 candidate_ids = torch.argmax(logits, dim=-1)
#                 probs = torch.softmax(logits, dim=-1)
#                 candidate_probs = torch.gather(
#                     probs, -1, candidate_ids.unsqueeze(-1)
#                 ).squeeze(-1)
#                 candidate_probs = (
#                     candidate_probs + gumbel_noise(candidate_probs) * current_temp
#                 )
#             else:
#                 probs = nn.functional.softmax(logits / temp, dim=-1)
#                 candidate_ids = torch.argmax(probs, dim=-1)  # [B, T_tgt]
#                 candidate_probs = torch.gather(
#                     probs, -1, candidate_ids.unsqueeze(-1)
#                 ).squeeze(-1)
#             ratio_mask = self.mask_prob(t + 1 / steps).item()
#             n_tokens_to_mask = int(T_tgt * ratio_mask)

#             if n_tokens_to_mask > 0:
#                 n_keep = T_tgt - n_tokens_to_mask
#                 _, keep_indices = torch.topk(candidate_probs, n_keep, dim=-1)
#                 mask_next = torch.ones_like(candidate_ids, dtype=torch.bool)
#                 mask_next.scatter_(1, keep_indices, False)
#                 new_target_part = candidate_ids.clone()
#                 new_target_part[mask_next] = self.mask_token
#                 # acoustic[:, T_prompt:] = new_target_part
#                 acoustic = new_target_part
#             else:
#                 # acoustic[:, T_prompt:] = candidate_ids
#                 acoustic = candidate_ids
#         # return acoustic[:, T_prompt:]
#         return acoustic
