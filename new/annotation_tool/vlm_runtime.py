"""
GPU/processor-specific runtime for frame-conditioned VLM generation.

This module intentionally isolates:
- processor-specific prompt packing
- device placement
- `generate()` invocation

Reasoning layers such as L2M/CoT and graph-based prompting should depend only
on the `FramePromptGenerator` interface so they can be tested without a GPU.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FramePromptGenerator(Protocol):
    """Minimal interface required by the reasoning pipeline."""

    def generate_text(
        self,
        frames: List[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> str:
        """Generate text from image frames and a prompt."""


class VLMGenerationRuntime:
    """Encapsulate multimodal prompt packing, device placement, and generation."""

    def __init__(
        self,
        model,
        processor,
        model_id: str,
        target_device: Optional[torch.device],
    ):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.target_device = target_device

    @property
    def is_qwen(self) -> bool:
        return "qwen" in self.model_id.lower()

    @property
    def uses_device_map(self) -> bool:
        device_map = getattr(self.model, "hf_device_map", None)
        return isinstance(device_map, dict) and len(device_map) > 0

    def _resolve_device_spec(self, device_spec) -> Optional[torch.device]:
        if isinstance(device_spec, torch.device):
            return device_spec

        if isinstance(device_spec, int):
            return torch.device(f"cuda:{device_spec}")

        if isinstance(device_spec, str):
            if device_spec in {"cpu", "mps", "cuda"}:
                return torch.device(device_spec)
            if device_spec.startswith(("cuda:", "mps:")):
                return torch.device(device_spec)

        return None

    def _devices_match(self, left: torch.device, right: torch.device) -> bool:
        if left == right:
            return True

        if left.type != right.type:
            return False

        if left.type == "cuda":
            left_index = 0 if left.index is None else left.index
            right_index = 0 if right.index is None else right.index
            return left_index == right_index

        return left.index == right.index

    @property
    def generation_device(self) -> Optional[torch.device]:
        if self.model is None:
            return self.target_device

        try:
            embedding = self.model.get_input_embeddings()
            if embedding is not None:
                if hasattr(embedding, "weight"):
                    return embedding.weight.device
                first_param = next(embedding.parameters(), None)
                if first_param is not None:
                    return first_param.device
        except Exception:
            pass

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for module_name in (
                "model.embed_tokens",
                "embed_tokens",
                "transformer.wte",
                "language_model.model.embed_tokens",
            ):
                if module_name in device_map:
                    resolved = self._resolve_device_spec(device_map[module_name])
                    if resolved is not None:
                        return resolved

            for mapped_device in device_map.values():
                resolved = self._resolve_device_spec(mapped_device)
                if resolved is not None:
                    return resolved

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.target_device

    def prepare_inputs_for_generation(self, inputs):
        target_device = self.generation_device
        if target_device is None:
            return inputs

        moved_inputs = inputs
        if hasattr(inputs, "to"):
            try:
                moved_inputs = inputs.to(target_device)
            except Exception as e:
                logger.warning(
                    "inputs.to(%s) failed during generation preparation: %s. "
                    "Falling back to recursive tensor move.",
                    target_device,
                    e,
                )

        moved_inputs = self._move_tensors_to_device(moved_inputs, target_device)

        off_device = sorted(
            {
                str(device)
                for device in self._collect_tensor_devices(moved_inputs)
                if device != target_device
            }
        )
        if off_device:
            raise RuntimeError(
                "Generation inputs were not fully moved to the target device. "
                f"target={target_device}, actual={off_device}"
            )

        return moved_inputs

    def _move_tensors_to_device(self, value, target_device):
        if torch.is_tensor(value):
            return value.to(target_device)

        if hasattr(value, "data") and isinstance(value.data, dict):
            value.data = {
                key: self._move_tensors_to_device(item, target_device)
                for key, item in value.data.items()
            }
            return value

        if isinstance(value, dict):
            return {
                key: self._move_tensors_to_device(item, target_device)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [self._move_tensors_to_device(item, target_device) for item in value]

        if isinstance(value, tuple):
            return tuple(self._move_tensors_to_device(item, target_device) for item in value)

        return value

    def _collect_tensor_devices(self, value):
        if torch.is_tensor(value):
            return [value.device]

        if hasattr(value, "data") and isinstance(value.data, dict):
            devices = []
            for item in value.data.values():
                devices.extend(self._collect_tensor_devices(item))
            return devices

        if isinstance(value, dict):
            devices = []
            for item in value.values():
                devices.extend(self._collect_tensor_devices(item))
            return devices

        if isinstance(value, (list, tuple)):
            devices = []
            for item in value:
                devices.extend(self._collect_tensor_devices(item))
            return devices

        return []

    def collect_model_devices(self):
        devices = set()

        if self.model is None:
            return devices

        for parameter in self.model.parameters():
            devices.add(parameter.device)

        for buffer in self.model.buffers():
            devices.add(buffer.device)

        return devices

    def validate_model_placement(self):
        if self.model is None or self.target_device is None:
            return

        devices = self.collect_model_devices()
        if not devices:
            logger.warning("Could not inspect model devices after loading.")
            return

        normalized_devices = {str(device) for device in devices}
        logger.info("Model parameter/buffer devices: %s", ", ".join(sorted(normalized_devices)))

        if len(devices) != 1 or not any(
            self._devices_match(self.target_device, device) for device in devices
        ):
            raise RuntimeError(
                "Model was not fully placed on a single target device. "
                f"target={self.target_device}, actual={sorted(normalized_devices)}"
            )

    def _build_inputs(self, frames: List[Image.Image], prompt: str):
        if self.is_qwen:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": frame} for frame in frames],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.processor(
                text=[text_prompt],
                images=frames,
                padding=True,
                return_tensors="pt",
            )

        return self.processor(
            images=frames,
            text=[prompt],
            return_tensors="pt",
            padding=True,
        )

    def generate_text(
        self,
        frames: List[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> str:
        inputs = self._build_inputs(frames, prompt)
        inputs = self.prepare_inputs_for_generation(inputs)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id

        logger.info(
            "Generating with %s frames, prompt length: %s chars",
            len(frames),
            len(prompt),
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if prompt in output_text:
            output_text = output_text.replace(prompt, "").strip()

        return output_text
