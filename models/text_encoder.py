import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Wraps a HuggingFace transformer (e.g. Qwen3-Embedding-8B) to produce
    per-token features for cross-attention fusion in the UNet decoder.

    Returns the backbone's raw hidden states -- no shared projection down to a
    fixed dim. Each CrossAttentionFusion (and VoxTell's PromptDecoder) has its
    own k_proj/v_proj/q_proj learning its own mapping from this raw hidden
    size, so those layers live in and train with the UNet regardless of
    embedding_cache (a single shared projection used to live only in this
    module, which is skipped entirely -- and never trained -- when
    load_backbone=False).

    forward() returns:
        hidden_states: (B, L, hidden_size) -- the backbone's own hidden size
        padding_mask:  (B, L) bool -- True at padding positions
                       (compatible with F.scaled_dot_product_attention additive bias)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        freeze_backbone: bool = True,
        finetune_last_n_layers: int = 0,
        load_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.transformer: nn.Module | None = None
        if not load_backbone:
            # Text features are precomputed (embedding_cache); forward() is never
            # called, so skip loading the multi-billion-param backbone entirely.
            return

        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.transformer.parameters():
                p.requires_grad_(False)
            if finetune_last_n_layers > 0:
                self._unfreeze_last_n_layers(finetune_last_n_layers)

    def _unfreeze_last_n_layers(self, n: int) -> None:
        """Unfreeze the last n transformer layers (works for both encoder and decoder models)."""
        layers = None
        # Try common attribute names across BERT-style and decoder-style models
        for attr in ("encoder.layer", "layers", "h", "decoder.layers"):
            obj = self.transformer
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__"):
                layers = obj
                break
        if layers is None:
            # Fallback: unfreeze via named_parameters on last-n named modules
            return
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        # Always unfreeze the final norm / head if present
        for attr in ("norm", "ln_f", "final_layer_norm"):
            m = getattr(self.transformer, attr, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L) — 1 for real tokens, 0 for padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        padding_mask = attention_mask.eq(0)  # (B, L) bool
        return outputs.last_hidden_state, padding_mask
