"""Load adapter files without silently changing their architecture variant.

Historical unmarked files belong to the original M architecture. M-Key files
must carry their variant and fixed position scale. Raw in-memory state dicts
retain the original low-level name/shape API; disk consumers use this helper.
"""
from __future__ import annotations

import math
import re

from safetensors import safe_open

from goqueen.fusion import GoQueen
from goqueen.m_key import MKeyGoQueen, POSITION_SCALE
from goqueen.assisted_point import AssistedPointGoQueen

APS_ROUTING_CONTRACT = "single_point_bool_mask_v1"


def load_adapter_checkpoint(model, checkpoint_path):
    """Validate one file's metadata and all shapes before copying any parameters.

    Only the exact known model types are supported. The same open file
    supplies metadata and tensors; no checkpoint path is reopened between
    those reads. Returns the file's metadata for callers that record it.
    """
    if type(model) is GoQueen:
        expected_variant = "m"
    elif type(model) is MKeyGoQueen:
        expected_variant = "m_key"
    elif type(model) is AssistedPointGoQueen:
        expected_variant = "m_assisted_point"
    else:
        raise ValueError("Adapter checkpoint loader requires an exact supported model type")

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as saved:
        metadata = saved.metadata() or {}
        variant = metadata.get("variant")
        if variant is None:
            if ("position_scale" in metadata or metadata.get("stage") == "a0_m_key"
                    or "routing_contract" in metadata or expected_variant == "m_assisted_point"):
                raise ValueError("Variant-specific checkpoint must explicitly declare its variant")
            variant = "m"
        elif variant not in ("m", "m_key", "m_assisted_point"):
            raise ValueError("Unknown or empty adapter checkpoint variant")
        if variant != expected_variant:
            raise ValueError("Adapter checkpoint variant does not match the constructed model")
        if variant == "m_key":
            try:
                scale = float(metadata["position_scale"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("M-Key checkpoint requires a finite fixed position_scale") from None
            if not math.isfinite(scale) or scale != POSITION_SCALE:
                raise ValueError("M-Key checkpoint position_scale does not match this implementation")
        elif "position_scale" in metadata or metadata.get("stage") == "a0_m_key":
            raise ValueError("Non-M-Key checkpoint metadata cannot declare an M-Key position scale or stage")

        if variant == "m_assisted_point":
            if metadata.get("routing_contract") != APS_ROUTING_CONTRACT:
                raise ValueError("APS checkpoint requires the exact routing contract")
            if metadata.get("engineering_only") not in ("true", "false"):
                raise ValueError("APS checkpoint must distinguish engineering from fitted weights")
            if not re.fullmatch(r"0|[1-9][0-9]*", metadata.get("updates", "")):
                raise ValueError("APS checkpoint requires a nonnegative completed update count")
            if metadata["engineering_only"] == "true" and metadata["updates"] != "0":
                raise ValueError("APS engineering weights must have zero completed optimizer updates")
            for name in ("source_manifest_sha256", "questions_sha256", "source_features_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", metadata.get(name, "")):
                    raise ValueError("APS checkpoint requires source and question digest metadata")
        elif "routing_contract" in metadata:
            raise ValueError("Non-APS checkpoints cannot declare an APS routing contract")

        parameters = {name: parameter for name, parameter in model.named_parameters()
                      if parameter.requires_grad}
        if set(saved.keys()) != set(parameters):
            raise ValueError("Adapter checkpoint parameter names do not match this architecture")
        state = {}
        for name, parameter in parameters.items():
            tensor = saved.get_tensor(name)
            if tensor.shape != parameter.shape:
                raise ValueError(f"Adapter checkpoint shape mismatch: {name}")
            if variant == "m_assisted_point" and not bool(tensor.isfinite().all()):
                raise ValueError("APS checkpoint parameters must be finite before any copying")
            state[name] = tensor
        model.load_adapter_state_dict(state)
    return dict(metadata)
