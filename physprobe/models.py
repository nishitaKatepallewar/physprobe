"""Frozen video encoders with hooks.

`Encoder` wraps a Hugging Face video transformer and exposes, per forward pass:
  * the residual stream after every block (mean-pooled, per-patch, per-time-step)
  * MLP hidden units (post-activation fc1) and MLP outputs (fc2)
  * attention edits (mask-and-renormalise ablations) and attention-distance stats
  * residual-stream interventions (for steering through the network)

V-JEPA 2 is the main target. VideoMAE (HF `VideoMAEModel`) is supported for the
residual/MLP hooks only, as a second architecture to compare against.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

ALIASES = {
    "vjepa2-l": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2-h": "facebook/vjepa2-vith-fpc64-256",
    "vjepa2-g": "facebook/vjepa2-vitg-fpc64-256",
    "vjepa2-g-384": "facebook/vjepa2-vitg-fpc64-384",
    "videomae-l": "MCG-NJU/videomae-large",
    "videomae-b": "MCG-NJU/videomae-base",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Capture:
    """What to keep from a forward pass. Layer lists index transformer blocks (0-based)."""

    pooled: bool = True  # [B, L, d]   mean over all tokens, every layer
    patch_layers: tuple = ()  # [B, G*G, d] mean over time, per spatial patch
    time_layers: tuple = ()  # [B, Tt, d]  mean over space, per temporal token
    fc1_layers: tuple = ()  # [B, 4d]     MLP hidden units (post-activation), mean over tokens
    fc2_layers: tuple = ()  # [B, d]      MLP output, mean over tokens
    token_layers: tuple = ()  # [B, N, d]   everything (large)


@dataclass
class AttnEdit:
    """Remove short-range attention at chosen layers (paper Sec. 6.3 / App. C.6).

    A query may not attend to keys that are 'near' it; the remaining attention is
    renormalised (implemented by masking logits before the softmax, which is the
    same thing). Near means:
        spatial only : patch distance <= s            (any time step)
        temporal only: |tubelet difference| <= t      (any patch)
        both given   : patch distance <= s AND |dt| <= t
    The paper does not say whether both-given means AND or OR, nor which heads are
    cut; `combine` and `heads` expose both choices. The default is OR: under AND the
    "combined" condition removes a subset of what "spatial s=7" removes, which cannot
    match the paper's finding that "combined" is by far the most damaging.
    """

    layers: tuple
    spatial: float | None = None
    temporal: int | None = None
    combine: str = "or"
    heads: tuple | None = None  # None = all heads


class Encoder:
    def __init__(self, model, kind: str, res: int, device="cpu", dtype=torch.float32,
                 name: str = ""):
        self.kind, self.res, self.device, self.dtype, self.name = kind, res, device, dtype, name
        self.model = model.to(device=device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = model.config
        if kind == "vjepa2":
            self.enc = model.encoder
            self.blocks = list(self.enc.layer)
            self.fc2 = [b.mlp.fc2 for b in self.blocks]
            self.patch = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
            self.tubelet = cfg.tubelet_size
            self.fixed_frames = None
            self._set_rope_grid(res // self.patch)
        elif kind == "videomae":
            self.enc = model
            self.blocks = list(model.encoder.layer)
            self.fc2 = [b.output.dense for b in self.blocks]
            self.patch, self.tubelet = cfg.patch_size, cfg.tubelet_size
            self.fixed_frames = cfg.num_frames
            if res != cfg.image_size:
                raise ValueError(f"VideoMAE needs res={cfg.image_size}")
        else:
            raise ValueError(kind)
        self.n_layers = len(self.blocks)
        self.d_model = cfg.hidden_size
        self.d_mlp = self.fc2[0].in_features
        self.n_heads = cfg.num_attention_heads
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    # ---------------------------------------------------------------- geometry
    def _set_rope_grid(self, g: int):
        """V-JEPA 2's rotary position ids are derived from `grid_size`, which the HF
        module fixes from config.crop_size (256 -> 16). Feeding another resolution
        (the paper uses 224 -> 14) without updating it silently mis-assigns every
        patch's row/column, so keep it in sync with the actual input (in the
        predictor too, when it is kept)."""
        for b in self.blocks:
            b.attention.grid_size = g
        pred = getattr(self.model, "predictor", None)
        for b in getattr(pred, "layer", []) or []:
            b.attention.grid_size = g
        self.G = g

    def grid(self, n_frames: int):
        return n_frames // self.tubelet, self.res // self.patch, self.res // self.patch

    def token_coords(self, n_frames: int):
        """(t, row, col) index of every token, in the model's token order."""
        Tt, G, _ = self.grid(n_frames)
        ids = torch.arange(Tt * G * G)
        return ids // (G * G), (ids % (G * G)) // G, ids % G

    def distances(self, n_frames: int):
        t, h, w = self.token_coords(n_frames)
        ds = torch.sqrt((h[:, None] - h[None]).float() ** 2 + (w[:, None] - w[None]).float() ** 2)
        dt = (t[:, None] - t[None]).abs().float()
        return ds, dt

    # ------------------------------------------------------------ preprocessing
    def preprocess(self, videos) -> torch.Tensor:
        """uint8 [B, T, H, W, 3] (numpy or tensor) -> normalised float [B, T, 3, res, res]."""
        x = torch.as_tensor(np.asarray(videos) if not torch.is_tensor(videos) else videos)
        x = x.to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0
        B, T, C, H, W = x.shape
        if (H, W) != (self.res, self.res):
            if H != W:  # centre crop to square first
                s = min(H, W)
                x = x[..., (H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
            x = F.interpolate(x.reshape(B * T, C, *x.shape[-2:]), size=(self.res, self.res),
                              mode="bilinear", antialias=True, align_corners=False)
            x = x.reshape(B, T, C, self.res, self.res)
        return ((x - self._mean) / self._std).to(self.dtype)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def run(self, videos, capture: Capture | None = None, attn_edit: AttnEdit | None = None,
            edits: dict | None = None, attn_stats: bool = False, preprocessed: bool = False):
        """One forward pass.

        edits: {layer: fn}, fn maps the block output [B, N, d] to a replacement; it is
               applied before capture, so later layers (and the capture) see the edit.
        Returns a dict of numpy arrays (see `Capture`), plus 'attn_spatial' /
        'attn_temporal' [L, heads] attention distance per head, SUMMED over the batch
        (divide by the number of videos), when attn_stats is set.
        """
        cap = capture or Capture()
        x = videos if preprocessed else self.preprocess(videos)
        B, T = x.shape[:2]
        if self.fixed_frames and T != self.fixed_frames:
            raise ValueError(f"{self.name} needs exactly {self.fixed_frames} frames, got {T}")
        Tt, G, _ = self.grid(T)
        out: dict = {k: {} for k in ("patch", "time", "fc1", "fc2", "tokens")}
        pooled = [None] * self.n_layers
        handles = []

        def block_hook(i):
            def hook(_m, _inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                new = None
                if edits and i in edits:
                    new = edits[i](h)
                    h = new
                hf = h.float()
                if cap.pooled:
                    pooled[i] = hf.mean(1).cpu()
                if i in cap.patch_layers or i in cap.time_layers:
                    g = hf.reshape(B, Tt, G * G, -1)
                    if i in cap.patch_layers:
                        out["patch"][i] = g.mean(1).cpu()
                    if i in cap.time_layers:
                        out["time"][i] = g.mean(2).cpu()
                if i in cap.token_layers:
                    out["tokens"][i] = hf.cpu()
                if new is not None:
                    if isinstance(output, tuple):
                        return (new,) + tuple(output[1:])
                    if isinstance(output, list):
                        return [new] + list(output[1:])
                    return new
            return hook

        for i, b in enumerate(self.blocks):
            handles.append(b.register_forward_hook(block_hook(i)))
        for i in set(cap.fc1_layers):
            handles.append(self.fc2[i].register_forward_pre_hook(
                lambda _m, inp, i=i: out["fc1"].__setitem__(i, inp[0].float().mean(1).cpu())))
        for i in set(cap.fc2_layers):
            handles.append(self.fc2[i].register_forward_hook(
                lambda _m, _inp, o, i=i: out["fc2"].__setitem__(i, o.float().mean(1).cpu())))

        stats = None
        with contextlib.ExitStack() as stack:
            stack.callback(lambda: [h.remove() for h in handles])
            if attn_edit is not None or attn_stats:
                stats = stack.enter_context(self._attention_taps(T, attn_edit, attn_stats))
            if self.kind == "vjepa2":
                self.enc(pixel_values_videos=x)
            else:
                self.enc(pixel_values=x)

        res = {}
        if cap.pooled:
            res["pooled"] = torch.stack(pooled, 1).numpy()
        for k in ("patch", "time", "fc1", "fc2", "tokens"):
            for i, v in out[k].items():
                res[f"{k}_L{i}"] = v.numpy()
        if attn_stats:
            res["attn_spatial"] = stats["spatial"].cpu().numpy()  # summed over the batch
            res["attn_temporal"] = stats["temporal"].cpu().numpy()
        return res

    # ---------------------------------------------------------------- predictor
    @torch.no_grad()
    def predict(self, videos, context_tubelets: int, edits: dict | None = None,
                preprocessed: bool = False):
        """Encoder pass (optionally with residual edits, as in `run`) followed by the
        V-JEPA 2 predictor: the encoder outputs of the first `context_tubelets`
        tubelets are the context, every later token is a prediction target.
        Returns float32 CPU tensors 'tokens' [B, N, d] (final encoder outputs, the
        prediction targets of an unedited pass) and 'pred' [B, N_target, d], plus the
        token indices 'target_ids'."""
        pred_mod = getattr(self.model, "predictor", None)
        if self.kind != "vjepa2" or pred_mod is None or isinstance(pred_mod, torch.nn.Identity):
            raise RuntimeError("load the encoder with keep_predictor=True (V-JEPA 2 only)")
        x = videos if preprocessed else self.preprocess(videos)
        B, T = x.shape[:2]
        Tt, G, _ = self.grid(T)
        if not 0 < context_tubelets < Tt:
            raise ValueError(f"context_tubelets must be in 1..{Tt - 1}")
        handles = []
        for i, fn in (edits or {}).items():
            def hook(_m, _inp, output, fn=fn):
                h = output[0] if isinstance(output, (tuple, list)) else output
                new = fn(h)
                if isinstance(output, tuple):
                    return (new,) + tuple(output[1:])
                if isinstance(output, list):
                    return [new] + list(output[1:])
                return new
            handles.append(self.blocks[i].register_forward_hook(hook))
        try:
            seq = self.enc(pixel_values_videos=x).last_hidden_state
        finally:
            for h in handles:
                h.remove()
        n_ctx = context_tubelets * G * G
        ids = torch.arange(Tt * G * G, device=seq.device)
        ctx, tgt = ids[:n_ctx].repeat(B, 1), ids[n_ctx:].repeat(B, 1)
        p = pred_mod(encoder_hidden_states=seq, context_mask=[ctx], target_mask=[tgt]).last_hidden_state
        return {"tokens": seq.float().cpu(), "pred": p.float().cpu(), "target_ids": ids[n_ctx:].cpu()}

    # ---------------------------------------------------------------- attention
    @contextlib.contextmanager
    def _attention_taps(self, n_frames: int, edit: AttnEdit | None, record: bool):
        """Swap in an explicit attention forward so that probabilities can be masked
        and measured. Without a mask it reproduces the stock module exactly
        (checked in `selftest`)."""
        if self.kind != "vjepa2":
            raise NotImplementedError("attention edits/stats are implemented for V-JEPA 2 only")
        ds, dt = (m.to(self.device) for m in self.distances(n_frames))
        mask = None
        if edit is not None:
            near_s = ds <= edit.spatial if edit.spatial is not None else None
            near_t = dt <= edit.temporal if edit.temporal is not None else None
            if near_s is not None and near_t is not None:
                mask = (near_s & near_t) if edit.combine == "and" else (near_s | near_t)
            else:
                mask = near_s if near_s is not None else near_t
            if mask is None:
                raise ValueError("AttnEdit needs spatial and/or temporal")
            dead = mask.all(1)  # a query with nothing left keeps itself
            mask[dead, torch.nonzero(dead).squeeze(1)] = False
        stats = {"spatial": torch.zeros(self.n_layers, self.n_heads, device=self.device),
                 "temporal": torch.zeros(self.n_layers, self.n_heads, device=self.device)}
        layers = range(self.n_layers) if record else (edit.layers if edit else ())
        originals = {}

        def make(i, attn):
            use_mask = mask if (edit is not None and i in edit.layers) else None

            def forward(hidden_states, position_mask=None, *args, **kwargs):
                B, N, _ = hidden_states.shape
                H, dh = attn.num_attention_heads, attn.attention_head_size
                q = attn.query(hidden_states).view(B, N, H, dh).transpose(1, 2)
                k = attn.key(hidden_states).view(B, N, H, dh).transpose(1, 2)
                v = attn.value(hidden_states).view(B, N, H, dh).transpose(1, 2)
                pos = attn.get_position_ids(hidden_states, masks=position_mask)
                k = attn.apply_rotary_embeddings(k, pos)
                q = attn.apply_rotary_embeddings(q, pos)
                logits = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling
                if use_mask is not None:
                    if edit.heads is None:
                        logits = logits.masked_fill(use_mask, float("-inf"))
                    else:
                        hs = list(edit.heads)
                        logits[:, hs] = logits[:, hs].masked_fill(use_mask, float("-inf"))
                probs = torch.softmax(logits.float(), dim=-1)
                if record:  # attention-weighted mean distance, averaged over queries
                    stats["spatial"][i] += (probs * ds).sum(-1).mean(-1).sum(0)
                    stats["temporal"][i] += (probs * dt).sum(-1).mean(-1).sum(0)
                probs = probs.to(q.dtype)
                ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, N, H * dh)
                return attn.proj(ctx), probs
            return forward

        try:
            for i in layers:
                attn = self.blocks[i].attention
                originals[i] = attn.forward
                attn.forward = make(i, attn)
            yield stats
        finally:
            for i, f in originals.items():
                # remove the instance attribute so the class method is used again
                del self.blocks[i].attention.forward


def load_encoder(name: str = "vjepa2-l", res: int = 224, device: str | None = None,
                 dtype: str = "auto", random_init: bool = False, tiny: bool = False,
                 keep_predictor: bool = False) -> Encoder:
    """Load a frozen encoder.

    name: alias in ALIASES, a Hugging Face repo id, or a local directory.
    random_init: build the architecture without downloading weights (pipeline
                 testing and the 'untrained network' control).
    tiny: with random_init, a 4-layer / 96-dim V-JEPA 2 (fast CPU smoke tests).
    keep_predictor: keep V-JEPA 2's predictor (for `Encoder.predict`); otherwise it is
                 dropped to save memory.
    """
    import transformers
    repo = ALIASES.get(name, name)
    kind = "videomae" if "videomae" in repo.lower() else "vjepa2"
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    td = {"auto": torch.float16 if device == "cuda" else torch.float32,
          "float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    if random_init:
        if kind == "vjepa2":
            sizes = {"vitl": (1024, 24, 16, 4.0), "vith": (1280, 32, 16, 4.0),
                     "vitg": (1408, 40, 22, 48 / 11)}
            hid, depth, heads, ratio = next((v for k, v in sizes.items() if k in repo), sizes["vitl"])
            if tiny:
                hid, depth, heads, ratio = 96, 4, 4, 4.0
            cfg = transformers.VJEPA2Config(
                hidden_size=hid, num_hidden_layers=depth, num_attention_heads=heads,
                mlp_ratio=ratio, crop_size=256, frames_per_clip=64,
                pred_hidden_size=48, pred_num_hidden_layers=1, pred_num_attention_heads=4)
            torch.manual_seed(0)
            model = transformers.VJEPA2Model(cfg)
        else:
            torch.manual_seed(0)
            model = transformers.VideoMAEModel(transformers.VideoMAEConfig(
                hidden_size=96, num_hidden_layers=4, num_attention_heads=4,
                intermediate_size=384) if tiny else transformers.VideoMAEConfig(
                hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                intermediate_size=4096))
        name = f"{name}-random"
    else:
        cls = transformers.VideoMAEModel if kind == "videomae" else transformers.AutoModel
        model = cls.from_pretrained(repo)
    if kind == "vjepa2" and hasattr(model, "predictor") and not keep_predictor:
        model.predictor = torch.nn.Identity()  # only the encoder is analysed
    return Encoder(model, kind, res, device, td, name=name)
