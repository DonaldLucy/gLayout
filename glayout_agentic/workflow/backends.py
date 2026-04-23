from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .skills import SkillMatch


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class BackendError(RuntimeError):
    pass


def ensure_runtime_identity_env() -> None:
    """Torch/Transformers can fail in containers whose UID has no passwd entry.

    In the Chipathon container this happens when Docker launches the process as the
    host UID/GID directly. `getpass.getuser()` then falls through to `pwd.getpwuid`
    and raises `KeyError`. We avoid that by setting a minimal identity/cache env
    before importing torch/transformers.
    """
    uid = os.getuid()
    username = (
        os.environ.get("LOGNAME")
        or os.environ.get("USER")
        or os.environ.get("LNAME")
        or os.environ.get("USERNAME")
        or f"uid{uid}"
    )
    os.environ.setdefault("LOGNAME", username)
    os.environ.setdefault("USER", username)
    os.environ.setdefault("LNAME", username)
    os.environ.setdefault("USERNAME", username)

    home = os.environ.get("HOME") or f"/tmp/codex-home-{uid}"
    os.environ.setdefault("HOME", home)

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path(home) / ".cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    torchinductor_dir = cache_root / "torchinductor"
    triton_dir = cache_root / "triton"
    hf_home = Path(os.environ.get("HF_HOME", cache_root / "huggingface"))

    torchinductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))


class BaseBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, skill_match: Optional[SkillMatch] = None) -> str:
        raise NotImplementedError


class SkillBackend(BaseBackend):
    def generate(self, prompt: str, skill_match: Optional[SkillMatch] = None) -> str:
        if skill_match is None:
            raise BackendError("Skill backend was selected, but no matching skill was found.")
        return skill_match.rendered_code


class LocalHFBackend(BaseBackend):
    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 1536,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        ensure_runtime_identity_env()
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise BackendError(
                "Local HF backend requires transformers, torch, and optional bitsandbytes."
            ) from exc

        load_kwargs: dict[str, object] = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **load_kwargs
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise BackendError(
                    "Adapter loading requires the `peft` package."
                ) from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)

        self._model = model
        self._tokenizer = tokenizer

    def generate(self, prompt: str, skill_match: Optional[SkillMatch] = None) -> str:
        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            raise BackendError("Model backend failed to initialize.")

        import torch

        messages = [
            {"role": "system", "content": "You write only Python source code."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt

        inputs = self._tokenizer(rendered_prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(generated, skip_special_tokens=True)
        return strip_code_fences(response)
