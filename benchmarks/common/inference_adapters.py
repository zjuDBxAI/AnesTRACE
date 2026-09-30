"""Local Transformers adapter for image and sampled-video benchmark inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ADAPTER_NAMES = ("auto", "transformers_vlm")


@dataclass(frozen=True)
class AdapterSpec:
    adapter_name: str
    model_type: str
    architectures: tuple[str, ...]


@dataclass(frozen=True)
class GenerationRequest:
    media_type: str
    media_path: Path
    system_prompt: str
    question_prompt: str
    media_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int | None
    generated_tokens: int | None


def thinking_template_kwargs(enabled: bool = False) -> dict[str, bool]:
    return {"enable_thinking": enabled}


def resolve_adapter_spec(model_path: Path, adapter_name: str = "auto") -> AdapterSpec:
    if adapter_name not in ADAPTER_NAMES:
        raise ValueError(f"unsupported adapter: {adapter_name}")
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return AdapterSpec(
        "transformers_vlm",
        str(config.get("model_type", "unknown")),
        tuple(config.get("architectures") or ()),
    )


class TransformersVLMAdapter:
    adapter_name = "transformers_vlm"

    def __init__(self, args: Any, spec: AdapterSpec):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the root requirements before local VLM inference") from exc
        self.torch = torch
        self.args = args
        self.model_type = spec.model_type
        path = str(Path(args.model_path).expanduser())
        dtype = getattr(args, "dtype", "auto")
        kwargs: dict[str, Any] = {"device_map": getattr(args, "device_map", "auto")}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, trust_remote_code=False, **kwargs
        )

    def _images(self, request: GenerationRequest) -> list[Any]:
        from PIL import Image

        paths = request.media_paths or (request.media_path,)
        if request.media_type == "image":
            return [Image.open(path).convert("RGB") for path in paths]
        if request.media_type != "video":
            raise ValueError(f"unsupported media type: {request.media_type}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video inference requires opencv-python") from exc
        cap = cv2.VideoCapture(str(request.media_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            count = min(int(getattr(self.args, "video_num_frames", 16)), total)
            if count <= 0:
                raise ValueError(f"video has no readable frames: {request.media_path}")
            frames = []
            for index in range(count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(index * (total - 1) / max(count - 1, 1)))
                ok, frame = cap.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not frames:
                raise ValueError(f"video decoding failed: {request.media_path}")
            return frames
        finally:
            cap.release()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        images = self._images(request)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": request.system_prompt}]},
            {"role": "user", "content": [
                *({"type": "image"} for _ in images),
                {"type": "text", "text": request.question_prompt},
            ]},
        ]
        template_args = dict(tokenize=False, add_generation_prompt=True)
        if getattr(self.args, "enable_thinking", False):
            template_args["enable_thinking"] = True
        try:
            prompt = self.processor.apply_chat_template(messages, **template_args)
        except TypeError:
            template_args.pop("enable_thinking", None)
            prompt = self.processor.apply_chat_template(messages, **template_args)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        length = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=bool(getattr(self.args, "temperature", 0.0)),
            )
        new_ids = ids[0, length:]
        return GenerationResult(
            self.processor.decode(new_ids, skip_special_tokens=True).strip(),
            length,
            int(new_ids.shape[-1]),
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [self.generate(request) for request in requests]


def create_adapter(args: Any, spec: AdapterSpec) -> TransformersVLMAdapter:
    return TransformersVLMAdapter(args, spec)
