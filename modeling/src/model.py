import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLM_CONFIGS

# Load default LLM profile from config
DEFAULT_PROFILE = LLM_CONFIGS["default"]

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time-series inputs.
    Normalizes per-instance statistics and applies learnable affine transform.
    """
    def __init__(self, num_series: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_series, 1))
        self.beta = nn.Parameter(torch.zeros(num_series, 1))

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps
        if mode == 'norm':
            return (x - mean) / std * self.gamma + self.beta
        elif mode == 'denorm':
            return (x - self.beta) / (self.gamma + self.eps) * std + mean
        else:
            raise ValueError(f"Unknown mode '{mode}' for RevIN. Use 'norm' or 'denorm'.")

class PatchEmbedder(nn.Module):
    """
    Splits each time series into patches and applies linear embedding.
    """
    def __init__(self, patch_len: int, stride: int, emb_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.linear = nn.Linear(patch_len, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        emb = self.linear(patches)
        return emb

class PatchReprogram(nn.Module):
    """
    Learns cross-attention between time-series patch embeddings and learnable text prototypes.
    """
    def __init__(self, patch_dim: int, model_dim: int, num_prototypes: int, num_heads: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, model_dim))
        self.Wq = nn.Linear(patch_dim, model_dim)
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(patch_emb)
        K = self.prototypes.unsqueeze(0).expand(Q.size(0), -1, -1)
        V = K
        out, _ = self.attn(Q, K, V)
        return out

class PromptPrefix(nn.Module):
    """
    Builds a prefix embedding from natural language prompt using a frozen LLM's input embeddings.
    """
    def __init__(self, prompt_llm: str = DEFAULT_PROFILE["prompt_llm"]):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(prompt_llm)
        self.text_embed = AutoModelForCausalLM.from_pretrained(prompt_llm).get_input_embeddings()
        for p in self.text_embed.parameters():
            p.requires_grad_(False)

    def build_prefix(self, domain_desc: str, task_inst: str, stats: str) -> torch.Tensor:
        prompt = (
            "<BOS> " + domain_desc.strip() + "\n"
            "### Task: " + task_inst.strip() + "\n"
            "### Stats: " + stats.strip() + " <EOS>\n"
        )
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
        tokens = tokens.to(self.text_embed.weight.device)
        with torch.no_grad():
            emb = self.text_embed(tokens)[0]
        return emb.squeeze(0)

class TimeLLM(nn.Module):
    """
    Full Time-LLM model: RevIN → PatchEmbedder → PatchReprogram → Prompt-as-Prefix → Frozen LLM → Output projection.
    """
    def __init__(
        self,
        prompt_llm: str = DEFAULT_PROFILE["prompt_llm"],
        backbone_llm: str = DEFAULT_PROFILE["backbone_llm"],
        num_series: int = 1,
        patch_len: int = 16,
        patch_stride: int = 16,
        patch_dim: int = 16,
        model_dim: int = 512,
        num_prototypes: int = 1000,
        num_heads: int = 8,
        pred_len: int = 96
    ):
        super().__init__()
        self.revin = RevIN(num_series)
        self.patch_embed = PatchEmbedder(patch_len, patch_stride, patch_dim)
        self.reprog = PatchReprogram(patch_dim, model_dim, num_prototypes, num_heads)
        self.prompt = PromptPrefix(prompt_llm)
        self.llm = AutoModelForCausalLM.from_pretrained(backbone_llm)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.output_proj = nn.Linear(model_dim, pred_len)

    def forward(self, x: torch.Tensor, domain_desc: str, task_inst: str, stats: str) -> torch.Tensor:
        B, N, T = x.shape
        x_norm = self.revin(x, mode='norm')
        outputs = []
        for i in range(N):
            xi = x_norm[:, i, :]
            patches = self.patch_embed(xi)
            reprog_out = self.reprog(patches)
            prefix_emb = self.prompt.build_prefix(domain_desc, task_inst, stats)
            llm_input = torch.cat([
                prefix_emb.unsqueeze(0).expand(B, -1, -1),
                reprog_out
            ], dim=1)
            llm_out = self.llm.transformer(inputs_embeds=llm_input).last_hidden_state
            patch_out = llm_out[:, prefix_emb.size(0):, :]
            pooled = patch_out.mean(dim=1)
            yhat = self.output_proj(pooled)
            outputs.append(yhat.unsqueeze(1))
        y_pred = torch.cat(outputs, dim=1)
        y_denorm = self.revin(y_pred, mode='denorm')
        return y_denorm
