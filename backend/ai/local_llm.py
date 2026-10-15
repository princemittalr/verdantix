"""Free local LLM fallback: Qwen2.5-1.5B-Instruct on CPU via transformers.

No key, no internet, no cost. Small models are unreliable at tool calling, so
the caller gathers facts from the twin first and the model only writes the
answer from those facts (grounded generation).
"""
import threading

import config as C

_lock = threading.Lock()
_model = None


def available():
    """True when the model weights are already on disk (never triggers a download)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(C.LOCAL_LLM, "config.json"), str)
    except Exception:
        return False


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(C.LOCAL_LLM, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(C.LOCAL_LLM, local_files_only=True, torch_dtype=torch.float32).eval()
        _model = (tok, model)
    return _model


def generate(system, user, max_new_tokens=350):
    import torch
    with _lock:
        tok, model = _load()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
