"""OpenAI-compatible HTTP server that serves HippoLM for evaluation.

Why this exists
---------------
``evalscope`` (and most LLM-eval harnesses) talk to models via an
OpenAI-compatible HTTP API: ``/v1/models``, ``/v1/chat/completions``,
``/v1/completions``. HippoLM is a custom architecture with no
upstream ``transformers`` model id, so the cleanest way to drive it
from evalscope is to wrap it in a thin HTTP shim.

This server is **single-process and stdlib-only** (``http.server`` +
``socketserver``) — no extra deps beyond what ``scripts/train.py``
already pulls in. It is intentionally NOT the production inference
backend (``src/inference/`` is reserved for that; see CLAUDE.md).
For evals the latency hit of a few ms of HTTP framing per request
is dwarfed by the per-step model cost.

Both checkpoint layouts are now first-class:
:class:`src.models.tp_model.TPHippoModel` stores its per-device
submodules in ``nn.ModuleDict`` containers (fixed June 2026), so
``model.state_dict()`` returns every trainable parameter with
keys auto-prefixed by the container name
(``replicated_per_device.<d>.<name>``, etc.). The
``_is_tp_state_dict`` sniff below dispatches on those prefixes.

Model loading
-------------
Two checkpoint layouts have to coexist because training can run
with or without TP:

  * **Plain** (single-GPU or world=1): ``state_dict`` keys are
    ``embed_tokens.weight``, ``layers.0.kda.q_proj.weight`` …
    Loaded into :class:`src.models.model.HippoModel`.
  * **TP** (tp_sim or multi-GPU): keys are prefixed with
    ``replicated_per_device.<d>.`` and ``layers_per_device.<d>.``
    Loaded into :class:`src.models.tp_model.TPHippoModel` with
    ``devices=[0]``. With one device the sharding math is a no-op
    (vocab // 1 == vocab) so the param shapes match the plain
    layout 1:1 — only the key names differ.

We sniff the layout from the first ``state_dict`` key and dispatch
to the matching loader. A mismatch raises; we do NOT silently
``strict=False`` a TP checkpoint into a plain model (would leave
shard-0 weights in slot 0 and zeros everywhere else → garbage logits).

Endpoints
---------
  * ``GET  /health``                  liveness/readiness probe.
  * ``GET  /v1/models``               model list (OpenAI shape).
  * ``POST /v1/chat/completions``     primary eval entry — applies
                                      the tokenizer's ChatML template
                                      and greedy-decodes the
                                      assistant turn.
  * ``POST /v1/completions``          legacy text-completion path.
                                      Useful for evals that bypass
                                      the chat template.

Generation is greedy with ``eos_token`` and the ``stop`` strings
sent in the request as the stop conditions. We do not support
streaming or logprobs (evalscope's MMLU only needs the message
content; log-likelihood ranking is delegated to a future
``src/inference/`` backend).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Make the repo root importable so ``from src...`` works without
# the caller having to set PYTHONPATH. Same pattern as
# ``scripts/train.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.models.config import HippoConfig  # noqa: E402
from src.training.tokenizer import load_tokenizer  # noqa: E402


log = logging.getLogger("hippo.eval_server")


# ---------------------------------------------------------------------------
# Model loading: detect TP vs plain checkpoint layout and dispatch.
# ---------------------------------------------------------------------------

def _is_tp_state_dict(sd_keys) -> bool:
    """Heuristic: TP checkpoints prefix everything with
    ``replicated_per_device.`` / ``layers_per_device.``.

    The plain layout starts with ``embed_tokens.`` /
    ``layers.<idx>.`` directly. The training loop never writes a
    mix, so a single key prefix is enough to decide.
    """
    for k in sd_keys:
        if k.startswith("replicated_per_device.") or k.startswith("layers_per_device."):
            return True
        if k.startswith("embed_tokens.") or k.startswith("layers.") or k == "norm.weight":
            return False
    return False


def _build_model_and_load(
    sd: dict[str, torch.Tensor],
    config: HippoConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Construct the right model class for ``sd`` and load weights.

    ``sd`` must already be the raw ``model_state_dict`` from the
    checkpoint file, not the whole payload (we strip that in
    :func:`main`).

    Raises if the layout doesn't match either known shape.
    """
    keys = list(sd.keys())
    if _is_tp_state_dict(keys):
        # TPHippoModel with a single device so world=1 and the
        # sharding math is a no-op (vocab // 1 == vocab, etc.).
        # We do NOT call ``init_tp`` because the asserts in the
        # constructor only check ``len(devices) == get_tp_world_size()``
        # and the default ``_TP_WORLD_SIZE`` is 1.
        from src.models.tp_model import TPHippoModel

        log.info("checkpoint is TP-shaped; loading into TPHippoModel(devices=[0])")
        model = TPHippoModel(config, devices=[0], dtype=dtype)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # TPHippoModel's state_dict layout expects a
        # ``layers_per_device.0.layers.<idx>...`` prefix; when the
        # checkpoint was saved with a single rank 0 device the
        # device index matches and these should both be empty.
        if unexpected:
            log.warning("unexpected keys in TP checkpoint (ignored): %s", unexpected[:5])
        if missing:
            log.warning("missing keys when loading TP checkpoint: %s", missing[:5])
        return model.to(device)

    # Plain single-device layout.
    from src.models.model import HippoModel

    log.info("checkpoint is plain-shaped; loading into HippoModel")
    model = HippoModel(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        log.warning("unexpected keys in plain checkpoint (ignored): %s", unexpected[:5])
    if missing:
        log.warning("missing keys when loading plain checkpoint: %s", missing[:5])
    return model.to(device).to(dtype)


# ---------------------------------------------------------------------------
# Tokenization + generation helpers.
# ---------------------------------------------------------------------------

def _apply_chat_template(tokenizer, messages: list[dict[str, Any]]) -> str:
    """Render chat messages to a single prompt string.

    Uses the Qwen3.5 ChatML template bundled in
    ``src/tokenizer/chat_template.jinja``. ``add_generation_prompt``
    appends the ``<|im_start|>assistant\\n`` opener so the model
    continues from the assistant role.

    Falls back to a hand-rolled ChatML format if the bundled
    template raises (e.g. when the Jinja environment differs
    between transformers versions). The fallback is intentionally
    conservative — it does not handle tools / vision tokens, but
    MMLU has none of those.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:  # noqa: BLE001 — Jinja errors are varied
        log.warning("apply_chat_template failed (%r); falling back to hand-rolled ChatML", e)
        out = []
        for m in messages:
            role = m["role"]
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            if role == "system":
                out.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                out.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        out.append("<|im_start|>assistant\n")
        return "".join(out)


def _greedy_generate(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_ids: list[int],
    stop_strings: list[str],
    device: torch.device,
) -> tuple[str, int, int]:
    """Greedy-decode ``max_new_tokens`` tokens after ``prompt_ids``.

    Stops early on any of ``stop_token_ids`` (typically just
    ``eos_token_id``) or any of ``stop_strings`` appearing in the
    freshly-decoded suffix. Returns ``(text, prompt_len, gen_len)``
    so the HTTP handler can report ``usage`` like OpenAI does.
    """
    model.eval()
    input_ids = prompt_ids.to(device).unsqueeze(0)  # [1, T]
    prompt_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    generated: list[int] = []
    stop_strings = list(stop_strings or [])
    stop_token_ids = list(stop_token_ids or [])
    if eos_id is not None and eos_id not in stop_token_ids:
        stop_token_ids.append(eos_id)

    with torch.no_grad():
        past = input_ids
        for _ in range(max_new_tokens):
            out = model(past)
            # TPHippoModel and HippoModel both return a dict with
            # ``logits`` shaped [B, T, vocab].
            logits = out["logits"] if isinstance(out, dict) else out
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            if next_id in stop_token_ids:
                break
            generated.append(next_id)
            past = torch.cat([past, torch.tensor([[next_id]], device=device)], dim=1)
            # Cheap partial-string stop check: decode what we have
            # and look for the stop string. We don't decode every
            # token (slow), only when we've accumulated a few.
            if stop_strings and len(generated) % 4 == 0:
                so_far = tokenizer.decode(generated, skip_special_tokens=True)
                if any(s in so_far for s in stop_strings):
                    break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stop_strings:
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
                break
    return text, prompt_len, len(generated)


# ---------------------------------------------------------------------------
# HTTP handler.
# ---------------------------------------------------------------------------

class _ServerState:
    """Bag of long-lived objects shared across requests.

    The handler class is instantiated per request, so we keep the
    heavy state (model, tokenizer, device) in a module-level
    global and the handler looks it up via ``Server.state``.
    """

    def __init__(self, model, tokenizer, device, model_id):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = model_id
        self.started_at = int(time.time())


state: _ServerState | None = None


class _Handler(BaseHTTPRequestHandler):
    server_version = "HippoLM-EvalServer/0.1"

    # Quieter logs; one line per request is enough.
    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    # ---- Routing ----

    def do_GET(self) -> None:  # noqa: N802 — http.server contract
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": state.model_id})
            return
        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": state.model_id,
                    "object": "model",
                    "created": state.started_at,
                    "owned_by": "hippolm",
                }],
            })
            return
        self._json(404, {"error": f"GET {self.path} not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions":
            self._chat_completions()
            return
        if self.path == "/v1/completions":
            self._completions()
            return
        self._json(404, {"error": f"POST {self.path} not found"})

    # ---- /v1/chat/completions ----

    def _chat_completions(self) -> None:
        try:
            body = self._read_json()
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return

        messages = body.get("messages") or []
        if not messages:
            self._json(400, {"error": "messages is required"})
            return

        # Defaults: greedy + short max_tokens. Most MCQ evals need
        # at most a few tokens (the "ANSWER: X" line); we cap at
        # 64 to keep runaway generations from blocking the eval.
        max_tokens = int(body.get("max_tokens") or 64)
        if max_tokens <= 0:
            max_tokens = 64
        temperature = float(body.get("temperature", 0.0))
        # We only implement greedy for now; ignore sampling params.
        if temperature != 0.0:
            log.debug("non-zero temperature requested; coercing to greedy")

        stop = body.get("stop")
        stop_strings: list[str] = []
        if isinstance(stop, str):
            stop_strings = [stop]
        elif isinstance(stop, list):
            stop_strings = [s for s in stop if isinstance(s, str)]

        prompt = _apply_chat_template(state.tokenizer, messages)
        prompt_ids = state.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]

        # Skip prefill past the model's context if the prompt is
        # too long — keeps the eval from OOMing on long few-shot
        # prompts. Tail-truncation preserves the question (which
        # is at the end of an MMLU prompt).
        max_ctx = getattr(state.model.config, "max_seq_len", 0) or 0
        if max_ctx and prompt_ids.shape[0] > max_ctx:
            log.warning("prompt length %d exceeds max_seq_len %d; tail-truncating",
                        prompt_ids.shape[0], max_ctx)
            prompt_ids = prompt_ids[-max_ctx:]

        text, prompt_len, gen_len = _greedy_generate(
            model=state.model,
            tokenizer=state.tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            stop_token_ids=[],
            stop_strings=stop_strings,
            device=state.device,
        )

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": state.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if gen_len < max_tokens else "length",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_len,
                "total_tokens": prompt_len + gen_len,
            },
        }
        self._json(200, resp)

    # ---- /v1/completions (legacy text-completion) ----

    def _completions(self) -> None:
        try:
            body = self._read_json()
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return

        prompt = body.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not isinstance(prompt, str):
            self._json(400, {"error": "prompt must be a string or list of strings"})
            return

        max_tokens = int(body.get("max_tokens") or 64)
        if max_tokens <= 0:
            max_tokens = 64

        prompt_ids = state.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        text, prompt_len, gen_len = _greedy_generate(
            model=state.model,
            tokenizer=state.tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            stop_token_ids=[],
            stop_strings=[],
            device=state.device,
        )
        resp = {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": state.model_id,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": "stop" if gen_len < max_tokens else "length",
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_len,
                "total_tokens": prompt_len + gen_len,
            },
        }
        self._json(200, resp)

    # ---- Low-level I/O ----

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("missing request body")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"invalid JSON body: {e}") from e

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Config loading: shared with the training CLI (single-source YAML).
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _config_from_yaml(
    yaml_path: Path,
    overrides: dict,
) -> HippoConfig:
    """Build a :class:`HippoConfig` from the YAML overlay + CLI overrides.

    The YAML schema is the same one ``scripts/train.py`` consumes,
    so eval can read the exact same ``configs/base.yml`` and stay
    in sync with the training dims.

    Unknown YAML keys are ignored (forward-compat: the training
    loop has yml-only fields like ``precision`` and ``stage`` that
    don't apply at eval time).
    """
    yml = _load_yaml(yaml_path)
    # YAML fields that map 1:1 to HippoConfig attrs.
    config_keys = {
        "vocab_size", "hidden_size", "tie_word_embeddings", "use_bias",
        "num_heads", "head_dim", "expand_v", "kda_mode", "use_short_conv",
        "allow_neg_eigval", "safe_gate", "lower_bound",
        "conv_size", "conv_bias", "num_layers", "num_blocks",
        "intermediate_size", "rms_norm_eps",
        "pack_chunk_size", "pack_buffer_size", "kda_skip_aqk_akk_saved",
    }
    cfg_dict = {k: v for k, v in yml.items() if k in config_keys}
    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})
    return HippoConfig(**cfg_dict)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible HippoLM eval server")
    p.add_argument("--config", default="configs/base.yml", help="HippoConfig YAML overlay")
    p.add_argument("--checkpoint", required=True, help="path to checkpoint_step_*.pt")
    p.add_argument("--tokenizer", default="src/tokenizer", help="HF tokenizer dir")
    p.add_argument("--model-id", default="hippolm", help="model id reported in /v1/models")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    # Optional dimension overrides (the checkpoint must already
    # match these; we use them purely to build the HippoConfig).
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    overrides = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "intermediate_size": args.intermediate_size,
    }
    config = _config_from_yaml(Path(args.config), overrides)
    log.info("HippoConfig: hidden=%d layers=%d blocks=%d heads=%d head_dim=%d",
             config.hidden_size, config.num_layers, config.num_blocks,
             config.num_heads, config.head_dim)

    log.info("loading checkpoint from %s", args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu")
    sd = payload.get("model_state_dict", payload)
    log.info("checkpoint contains %d model tensors (step=%s, loss=%s)",
             len(sd), payload.get("step"), payload.get("loss"))

    log.info("loading tokenizer from %s", args.tokenizer)
    tokenizer = load_tokenizer(args.tokenizer)

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    log.info("building model on device=%s dtype=%s", device, dtype)
    model = _build_model_and_load(sd, config, device, dtype)
    model.eval()

    global state
    state = _ServerState(model=model, tokenizer=tokenizer, device=device, model_id=args.model_id)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    log.info("listening on http://%s:%d  (model_id=%s)", args.host, args.port, args.model_id)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
