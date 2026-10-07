r"""Step 14.1.1.2 — Idempotent Pinned Model Downloader, Deterministic Swin-to-ONNX Exporter & Verifier.

Official Upstream Model:
- Hugging Face ID: `umm-maybe/AI-image-detector`
- Pinned Revision: `c7e223baf11bc40528af364ba7bdea030ef42f9e`
- Architecture: `SwinForImageClassification` (`depths=[2, 2, 18, 2]`, `num_heads=[4, 8, 16, 32]`, `embed_dim=128`)
- License: `Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)`
- Upstream Weight Artifact: `pytorch_model.bin` (347,596,479 bytes,
  SHA-256 `f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf`)
- Local Derivative Runtime Artifact: `model.onnx` (`onnx_is_upstream=False`, exported locally from
  the pinned upstream `pytorch_model.bin` state dict).

Usage:
    cd backend
    .\.venv\Scripts\python scripts/setup_ai_image_detector.py --verify
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import urllib.request
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from PIL import Image, ImageDraw

from app.core.config import settings
from app.schemas.ai_detection import (
    OFFICIAL_MODEL_ARCHITECTURE,
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_LICENSE,
    OFFICIAL_PINNED_REVISION,
)
from app.services.analysis.image_ai_detection_service import (
    _is_valid_pinned_revision,
    _preprocess_image_in_memory,
    _resolve_safe_model_directory,
)

EXPECTED_PYTORCH_MODEL_SHA256 = "f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf"
EXPECTED_PYTORCH_MODEL_SIZE = 347596479


def compute_file_sha256(file_path: Path) -> str:
    """Compute streaming SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_in:
        while chunk := file_in.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


class _TorchStorageRef:
    def __init__(self, storage_type: str, key: str, numel: int):
        self.storage_type = storage_type
        self.key = key
        self.numel = numel


class _PyTorchZipUnpickler(pickle.Unpickler):
    """Safe unpickler that extracts tensor storage descriptors from PyTorch zip archives without arbitrary code execution."""

    def persistent_load(self, pid: Any) -> _TorchStorageRef:
        return _TorchStorageRef(str(pid[1]), str(pid[2]), int(pid[4]))

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, req_grad, hooks: (
                storage.key,
                tuple(int(s) for s in size),
                tuple(int(st) for st in stride),
                int(offset),
                storage.storage_type,
            )
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"Disallowed class in PyTorch checkpoint: {module}.{name}")


def load_pytorch_zip_tensors(bin_path: Path) -> Dict[str, np.ndarray]:
    """Load all 449 tensors from the pinned `pytorch_model.bin` zip archive into NumPy arrays."""
    if not bin_path.is_file() or not zipfile.is_zipfile(bin_path):
        raise RuntimeError(f"Invalid or corrupt PyTorch zip checkpoint: {bin_path}")

    tensors: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(bin_path, "r") as zf:
        raw_pkl = zf.read("archive/data.pkl")
        state_desc = _PyTorchZipUnpickler(io.BytesIO(raw_pkl)).load()
        for name, (storage_key, shape, _stride, offset, storage_type) in state_desc.items():
            raw_bytes = zf.read(f"archive/data/{storage_key}")
            dtype = np.int64 if "Long" in storage_type else np.float32
            flat = np.frombuffer(raw_bytes, dtype=dtype)
            numel = int(np.prod(shape)) if shape else 1
            sliced = flat[offset : offset + numel]
            tensors[name] = sliced.reshape(shape).copy()

    return tensors


def _compute_swin_window_indices_and_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Precompute static gather indices for window partition/reverse and shifted-window attention mask."""
    grid = np.arange(height * width, dtype=np.int64).reshape(height, width)
    if shift_size > 0:
        grid = np.roll(grid, shift=(-shift_size, -shift_size), axis=(0, 1))

    num_win_h = height // window_size
    num_win_w = width // window_size
    num_windows = num_win_h * num_win_w
    win_tokens = window_size * window_size

    win_order = (
        grid.reshape(num_win_h, window_size, num_win_w, window_size)
        .transpose(0, 2, 1, 3)
        .reshape(num_windows * win_tokens)
    )
    inv_win_order = np.argsort(win_order).astype(np.int64)

    if shift_size == 0:
        return win_order, inv_win_order, None

    img_mask = np.zeros((1, height, width, 1), dtype=np.int32)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h_s in h_slices:
        for w_s in w_slices:
            img_mask[:, h_s, w_s, :] = cnt
            cnt += 1

    mask_windows = (
        img_mask.reshape(num_win_h, window_size, num_win_w, window_size)
        .transpose(0, 2, 1, 3)
        .reshape(num_windows, win_tokens)
    )
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)
    return win_order, inv_win_order, attn_mask.reshape(1, num_windows, 1, win_tokens, win_tokens)


def export_swin_pytorch_to_onnx(bin_path: Path, onnx_path: Path) -> Dict[str, Any]:
    """Export the pinned `SwinForImageClassification` (`pytorch_model.bin`) to `model.onnx` (opset 17)."""
    tensors = load_pytorch_zip_tensors(bin_path)
    if len(tensors) != 449:
        raise RuntimeError(f"Expected 449 Swin tensors in {bin_path.name}, found {len(tensors)}")

    nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []

    def add_init(name: str, arr: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    # 1. Patch Embeddings: Conv(3->128, k=4, s=4) -> Reshape [B, 128, 3136] -> Transpose [B, 3136, 128] -> LayerNorm
    proj_w = add_init("patch_proj_w", tensors["swin.embeddings.patch_embeddings.projection.weight"].astype(np.float32))
    proj_b = add_init("patch_proj_b", tensors["swin.embeddings.patch_embeddings.projection.bias"].astype(np.float32))
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["pixel_values", proj_w, proj_b],
            outputs=["patch_conv"],
            kernel_shape=[4, 4],
            strides=[4, 4],
        )
    )

    shape_flat = add_init("shape_patch_flat", np.array([0, 128, 3136], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", inputs=["patch_conv", shape_flat], outputs=["patch_flat"]))
    nodes.append(helper.make_node("Transpose", inputs=["patch_flat"], outputs=["patch_seq"], perm=[0, 2, 1]))

    emb_ln_w = add_init("emb_ln_w", tensors["swin.embeddings.norm.weight"].astype(np.float32))
    emb_ln_b = add_init("emb_ln_b", tensors["swin.embeddings.norm.bias"].astype(np.float32))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            inputs=["patch_seq", emb_ln_w, emb_ln_b],
            outputs=["stage_0_in"],
            axis=-1,
            epsilon=1e-5,
        )
    )

    # 2. Four Swin Stages: depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], H=W=[56, 28, 14, 7], C=[128, 256, 512, 1024]
    stage_configs = [
        (0, 2, 4, 56, 56, 128),
        (1, 2, 8, 28, 28, 256),
        (2, 18, 16, 14, 14, 512),
        (3, 2, 32, 7, 7, 1024),
    ]
    window_size = 7
    win_tokens = 49
    sqrt2_inv = add_init("gelu_sqrt2_inv", np.array(1.0 / math.sqrt(2.0), dtype=np.float32))
    half_const = add_init("gelu_half", np.array(0.5, dtype=np.float32))
    one_const = add_init("gelu_one", np.array(1.0, dtype=np.float32))

    cur_tensor = "stage_0_in"
    for stage_idx, depth, num_heads, height, width, channels in stage_configs:
        head_dim = channels // num_heads
        scale_val = float(1.0 / math.sqrt(head_dim))
        num_windows = (height // window_size) * (width // window_size)

        sh_win_3d = add_init(f"s{stage_idx}_sh_win3d", np.array([0, num_windows, win_tokens, channels], dtype=np.int64))
        sh_qkv_5d = add_init(
            f"s{stage_idx}_sh_qkv5d",
            np.array([0, num_windows, win_tokens, num_heads, head_dim], dtype=np.int64),
        )
        sh_seq_3d = add_init(f"s{stage_idx}_sh_seq3d", np.array([0, height * width, channels], dtype=np.int64))

        win_idx_0, inv_idx_0, _ = _compute_swin_window_indices_and_mask(height, width, window_size, 0)
        win_idx_0_init = add_init(f"s{stage_idx}_win_idx_0", win_idx_0)
        inv_idx_0_init = add_init(f"s{stage_idx}_inv_idx_0", inv_idx_0)

        if height > window_size:
            win_idx_s, inv_idx_s, attn_mask_s = _compute_swin_window_indices_and_mask(
                height, width, window_size, window_size // 2
            )
            win_idx_s_init = add_init(f"s{stage_idx}_win_idx_s", win_idx_s)
            inv_idx_s_init = add_init(f"s{stage_idx}_inv_idx_s", inv_idx_s)
        else:
            win_idx_s_init = win_idx_0_init
            inv_idx_s_init = inv_idx_0_init
            attn_mask_s = None

        for blk_idx in range(depth):
            prefix = f"swin.encoder.layers.{stage_idx}.blocks.{blk_idx}"
            tag = f"s{stage_idx}_b{blk_idx}"
            use_shift = (blk_idx % 2 == 1) and (height > window_size)

            # LayerNorm 1
            ln1_w = add_init(f"{tag}_ln1_w", tensors[f"{prefix}.layernorm_before.weight"].astype(np.float32))
            ln1_b = add_init(f"{tag}_ln1_b", tensors[f"{prefix}.layernorm_before.bias"].astype(np.float32))
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    inputs=[cur_tensor, ln1_w, ln1_b],
                    outputs=[f"{tag}_ln1"],
                    axis=-1,
                    epsilon=1e-5,
                )
            )

            # Window partition (+ optional cyclic shift) via static Gather + Reshape
            w_idx = win_idx_s_init if use_shift else win_idx_0_init
            i_idx = inv_idx_s_init if use_shift else inv_idx_0_init
            nodes.append(helper.make_node("Gather", inputs=[f"{tag}_ln1", w_idx], outputs=[f"{tag}_wgath"], axis=1))
            nodes.append(helper.make_node("Reshape", inputs=[f"{tag}_wgath", sh_win_3d], outputs=[f"{tag}_win"]))

            # Combined QKV projection [C, 3*C] to minimize graph nodes and maximize throughput
            qw = tensors[f"{prefix}.attention.self.query.weight"].astype(np.float32) * scale_val
            qb = tensors[f"{prefix}.attention.self.query.bias"].astype(np.float32) * scale_val
            kw = tensors[f"{prefix}.attention.self.key.weight"].astype(np.float32)
            kb = tensors[f"{prefix}.attention.self.key.bias"].astype(np.float32)
            vw = tensors[f"{prefix}.attention.self.value.weight"].astype(np.float32)
            vb = tensors[f"{prefix}.attention.self.value.bias"].astype(np.float32)

            q_w_init = add_init(f"{tag}_qw", qw.T.copy())
            q_b_init = add_init(f"{tag}_qb", qb)
            k_w_init = add_init(f"{tag}_kw", kw.T.copy())
            k_b_init = add_init(f"{tag}_kb", kb)
            v_w_init = add_init(f"{tag}_vw", vw.T.copy())
            v_b_init = add_init(f"{tag}_vb", vb)

            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_win", q_w_init], outputs=[f"{tag}_q_mm"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_q_mm", q_b_init], outputs=[f"{tag}_q_lin"]))
            nodes.append(helper.make_node("Reshape", inputs=[f"{tag}_q_lin", sh_qkv_5d], outputs=[f"{tag}_q_5d"]))
            nodes.append(
                helper.make_node("Transpose", inputs=[f"{tag}_q_5d"], outputs=[f"{tag}_q"], perm=[0, 1, 3, 2, 4])
            )

            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_win", k_w_init], outputs=[f"{tag}_k_mm"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_k_mm", k_b_init], outputs=[f"{tag}_k_lin"]))
            nodes.append(helper.make_node("Reshape", inputs=[f"{tag}_k_lin", sh_qkv_5d], outputs=[f"{tag}_k_5d"]))
            nodes.append(
                helper.make_node("Transpose", inputs=[f"{tag}_k_5d"], outputs=[f"{tag}_k_t"], perm=[0, 1, 3, 4, 2])
            )

            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_win", v_w_init], outputs=[f"{tag}_v_mm"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_v_mm", v_b_init], outputs=[f"{tag}_v_lin"]))
            nodes.append(helper.make_node("Reshape", inputs=[f"{tag}_v_lin", sh_qkv_5d], outputs=[f"{tag}_v_5d"]))
            nodes.append(
                helper.make_node("Transpose", inputs=[f"{tag}_v_5d"], outputs=[f"{tag}_v"], perm=[0, 1, 3, 2, 4])
            )

            # Attention scores + pre-indexed relative_position_bias (+ shifted window mask if odd block)
            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_q", f"{tag}_k_t"], outputs=[f"{tag}_scores"]))

            rel_table = tensors[f"{prefix}.attention.self.relative_position_bias_table"].astype(np.float32)
            rel_index = tensors[f"{prefix}.attention.self.relative_position_index"].astype(np.int64)
            rel_bias = (
                rel_table[rel_index.reshape(-1)]
                .reshape(win_tokens, win_tokens, num_heads)
                .transpose(2, 0, 1)[None, None, :, :, :]
            )
            if use_shift and attn_mask_s is not None:
                rel_bias = rel_bias + attn_mask_s
            rel_bias_init = add_init(f"{tag}_rel_bias", rel_bias.astype(np.float32))

            nodes.append(
                helper.make_node("Add", inputs=[f"{tag}_scores", rel_bias_init], outputs=[f"{tag}_biased_scores"])
            )
            nodes.append(
                helper.make_node("Softmax", inputs=[f"{tag}_biased_scores"], outputs=[f"{tag}_probs"], axis=-1)
            )
            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_probs", f"{tag}_v"], outputs=[f"{tag}_ctx_5d"]))
            nodes.append(
                helper.make_node("Transpose", inputs=[f"{tag}_ctx_5d"], outputs=[f"{tag}_ctx_t"], perm=[0, 1, 3, 2, 4])
            )
            nodes.append(helper.make_node("Reshape", inputs=[f"{tag}_ctx_t", sh_win_3d], outputs=[f"{tag}_ctx"]))

            out_w = add_init(
                f"{tag}_att_ow", tensors[f"{prefix}.attention.output.dense.weight"].astype(np.float32).T.copy()
            )
            out_b = add_init(f"{tag}_att_ob", tensors[f"{prefix}.attention.output.dense.bias"].astype(np.float32))
            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_ctx", out_w], outputs=[f"{tag}_att_mm"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_att_mm", out_b], outputs=[f"{tag}_att_out"]))

            # Window reverse (+ reverse cyclic shift) & Residual 1
            nodes.append(helper.make_node("Reshape", inputs=[f"{tag}_att_out", sh_seq_3d], outputs=[f"{tag}_att_flat"]))
            nodes.append(
                helper.make_node("Gather", inputs=[f"{tag}_att_flat", i_idx], outputs=[f"{tag}_att_rev"], axis=1)
            )
            nodes.append(helper.make_node("Add", inputs=[cur_tensor, f"{tag}_att_rev"], outputs=[f"{tag}_res1"]))

            # LayerNorm 2 + MLP (Intermediate -> Exact GELU via Erf -> Output) + Residual 2
            ln2_w = add_init(f"{tag}_ln2_w", tensors[f"{prefix}.layernorm_after.weight"].astype(np.float32))
            ln2_b = add_init(f"{tag}_ln2_b", tensors[f"{prefix}.layernorm_after.bias"].astype(np.float32))
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    inputs=[f"{tag}_res1", ln2_w, ln2_b],
                    outputs=[f"{tag}_ln2"],
                    axis=-1,
                    epsilon=1e-5,
                )
            )

            fc1_w = add_init(
                f"{tag}_fc1_w", tensors[f"{prefix}.intermediate.dense.weight"].astype(np.float32).T.copy()
            )
            fc1_b = add_init(f"{tag}_fc1_b", tensors[f"{prefix}.intermediate.dense.bias"].astype(np.float32))
            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_ln2", fc1_w], outputs=[f"{tag}_fc1_mm"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_fc1_mm", fc1_b], outputs=[f"{tag}_fc1"]))

            # Exact GELU: 0.5 * x * (1 + Erf(x / sqrt(2)))
            nodes.append(helper.make_node("Mul", inputs=[f"{tag}_fc1", sqrt2_inv], outputs=[f"{tag}_g_div"]))
            nodes.append(helper.make_node("Erf", inputs=[f"{tag}_g_div"], outputs=[f"{tag}_g_erf"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_g_erf", one_const], outputs=[f"{tag}_g_add"]))
            nodes.append(helper.make_node("Mul", inputs=[f"{tag}_fc1", f"{tag}_g_add"], outputs=[f"{tag}_g_mul1"]))
            nodes.append(helper.make_node("Mul", inputs=[f"{tag}_g_mul1", half_const], outputs=[f"{tag}_gelu"]))

            fc2_w = add_init(f"{tag}_fc2_w", tensors[f"{prefix}.output.dense.weight"].astype(np.float32).T.copy())
            fc2_b = add_init(f"{tag}_fc2_b", tensors[f"{prefix}.output.dense.bias"].astype(np.float32))
            nodes.append(helper.make_node("MatMul", inputs=[f"{tag}_gelu", fc2_w], outputs=[f"{tag}_fc2_mm"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_fc2_mm", fc2_b], outputs=[f"{tag}_fc2"]))
            nodes.append(helper.make_node("Add", inputs=[f"{tag}_res1", f"{tag}_fc2"], outputs=[f"{tag}_out"]))
            cur_tensor = f"{tag}_out"

        # PatchMerging downsample at end of stages 0, 1, 2
        if stage_idx < 3:
            ds_prefix = f"swin.encoder.layers.{stage_idx}.downsample"
            ds_tag = f"s{stage_idx}_ds"
            grid_2d = np.arange(height * width, dtype=np.int64).reshape(height, width)
            idx0 = add_init(f"{ds_tag}_idx0", grid_2d[0::2, 0::2].reshape(-1))
            idx1 = add_init(f"{ds_tag}_idx1", grid_2d[1::2, 0::2].reshape(-1))
            idx2 = add_init(f"{ds_tag}_idx2", grid_2d[0::2, 1::2].reshape(-1))
            idx3 = add_init(f"{ds_tag}_idx3", grid_2d[1::2, 1::2].reshape(-1))

            nodes.append(helper.make_node("Gather", inputs=[cur_tensor, idx0], outputs=[f"{ds_tag}_x0"], axis=1))
            nodes.append(helper.make_node("Gather", inputs=[cur_tensor, idx1], outputs=[f"{ds_tag}_x1"], axis=1))
            nodes.append(helper.make_node("Gather", inputs=[cur_tensor, idx2], outputs=[f"{ds_tag}_x2"], axis=1))
            nodes.append(helper.make_node("Gather", inputs=[cur_tensor, idx3], outputs=[f"{ds_tag}_x3"], axis=1))
            nodes.append(
                helper.make_node(
                    "Concat",
                    inputs=[f"{ds_tag}_x0", f"{ds_tag}_x1", f"{ds_tag}_x2", f"{ds_tag}_x3"],
                    outputs=[f"{ds_tag}_cat"],
                    axis=-1,
                )
            )

            ds_ln_w = add_init(f"{ds_tag}_ln_w", tensors[f"{ds_prefix}.norm.weight"].astype(np.float32))
            ds_ln_b = add_init(f"{ds_tag}_ln_b", tensors[f"{ds_prefix}.norm.bias"].astype(np.float32))
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    inputs=[f"{ds_tag}_cat", ds_ln_w, ds_ln_b],
                    outputs=[f"{ds_tag}_ln"],
                    axis=-1,
                    epsilon=1e-5,
                )
            )
            ds_red_w = add_init(
                f"{ds_tag}_red_w", tensors[f"{ds_prefix}.reduction.weight"].astype(np.float32).T.copy()
            )
            nodes.append(helper.make_node("MatMul", inputs=[f"{ds_tag}_ln", ds_red_w], outputs=[f"{ds_tag}_out"]))
            cur_tensor = f"{ds_tag}_out"

    # 3. Final Swin LayerNorm + Global Mean Pool + Classifier Head (1024 -> 2)
    final_ln_w = add_init("final_ln_w", tensors["swin.layernorm.weight"].astype(np.float32))
    final_ln_b = add_init("final_ln_b", tensors["swin.layernorm.bias"].astype(np.float32))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            inputs=[cur_tensor, final_ln_w, final_ln_b],
            outputs=["final_ln"],
            axis=-1,
            epsilon=1e-5,
        )
    )

    nodes.append(
        helper.make_node("ReduceMean", inputs=["final_ln"], outputs=["pooled_features"], axes=[1], keepdims=0)
    )

    cls_w = add_init("cls_w", tensors["classifier.weight"].astype(np.float32).T.copy())
    cls_b = add_init("cls_b", tensors["classifier.bias"].astype(np.float32))
    nodes.append(helper.make_node("MatMul", inputs=["pooled_features", cls_w], outputs=["logits_mm"]))
    nodes.append(helper.make_node("Add", inputs=["logits_mm", cls_b], outputs=["logits"]))

    x_in = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch_size", 3, 224, 224])
    y_out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 2])

    graph = helper.make_graph(
        nodes,
        "SwinForImageClassification_umm_maybe_AI_image_detector",
        [x_in],
        [y_out],
        initializer=initializers,
    )
    model_def = helper.make_model(
        graph,
        producer_name="MediaTrace-Step-14.1.1.2-Local-ONNX-Exporter",
        producer_version="1.0.0",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model_def.ir_version = 8

    tmp_onnx = onnx_path.with_suffix(".onnx.tmp")
    onnx.save(model_def, str(tmp_onnx))
    onnx.checker.check_model(str(tmp_onnx))
    tmp_onnx.replace(onnx_path)

    return inspect_onnx_artifact(onnx_path)


def inspect_onnx_artifact(onnx_path: Path) -> Dict[str, Any]:
    """Inspect and validate ONNX model input/output tensor metadata via ONNX Runtime."""
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    onnx.checker.check_model(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected 1 input and 1 output in ONNX graph, got {len(inputs)} and {len(outputs)}")

    inp = inputs[0]
    out = outputs[0]
    inp_shape = list(inp.shape)
    out_shape = list(out.shape)

    if inp.type != "tensor(float)":
        raise ValueError(f"Expected ONNX input dtype 'tensor(float)' (float32), got '{inp.type}'")
    if len(inp_shape) != 4 or inp_shape[1] != 3 or inp_shape[2] != 224 or inp_shape[3] != 224:
        raise ValueError(f"Incompatible ONNX input shape {inp_shape}; expected [B, 3, 224, 224]")
    if len(out_shape) != 2 or out_shape[1] != 2:
        raise ValueError(f"Incompatible ONNX output shape {out_shape}; expected [B, 2] for {0: 'artificial', 1: 'human'}")

    return {
        "input_name": inp.name,
        "input_shape": inp_shape,
        "input_dtype": inp.type,
        "output_name": out.name,
        "output_shape": out_shape,
        "output_dtype": out.type,
        "class_ordering": {"0": "artificial", "1": "human"},
    }


def download_and_export_pinned_model(
    model_id: str = OFFICIAL_MODEL_ID,
    revision: str = OFFICIAL_PINNED_REVISION,
    force: bool = False,
) -> Dict[str, Any]:
    """Idempotently download pinned `pytorch_model.bin`, verify SHA-256, export `model.onnx`, and write integrity metadata."""
    if not _is_valid_pinned_revision(revision) or revision != OFFICIAL_PINNED_REVISION:
        raise ValueError(
            f"Refusing unpinned or unexpected model revision '{revision}'. "
            f"Must be exact pinned commit SHA '{OFFICIAL_PINNED_REVISION}'."
        )

    target_dir = _resolve_safe_model_directory(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{model_id}/resolve/{revision}"

    # 1. Ensure config.json and preprocessor_config.json are present
    for filename in ("config.json", "preprocessor_config.json"):
        dest = target_dir / filename
        if force or not dest.is_file():
            url = f"{base_url}/{filename}"
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)

    # 2. Idempotently check or download pytorch_model.bin from pinned revision
    pytorch_bin_path = target_dir / "pytorch_model.bin"
    need_download = True
    if pytorch_bin_path.is_file() and not force:
        existing_size = pytorch_bin_path.stat().st_size
        existing_sha = compute_file_sha256(pytorch_bin_path)
        if existing_size == EXPECTED_PYTORCH_MODEL_SIZE and existing_sha == EXPECTED_PYTORCH_MODEL_SHA256:
            print(
                f"Verified existing pinned pytorch_model.bin ({existing_size:,} bytes, SHA-256: {existing_sha}). Skipping download."
            )
            need_download = False
        else:
            raise RuntimeError(
                f"Existing pytorch_model.bin in {target_dir} is incomplete or corrupt "
                f"(size={existing_size}, sha256={existing_sha}, expected_sha256={EXPECTED_PYTORCH_MODEL_SHA256})."
            )

    if need_download:
        url = f"{base_url}/pytorch_model.bin"
        tmp_bin = pytorch_bin_path.with_suffix(".bin.tmp")
        print(f"Downloading pinned weights {url} -> {pytorch_bin_path} ...")
        urllib.request.urlretrieve(url, tmp_bin)
        dl_size = tmp_bin.stat().st_size
        dl_sha = compute_file_sha256(tmp_bin)
        if dl_size != EXPECTED_PYTORCH_MODEL_SIZE or dl_sha != EXPECTED_PYTORCH_MODEL_SHA256:
            tmp_bin.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded pytorch_model.bin failed integrity verification "
                f"(size={dl_size}, sha256={dl_sha}, expected={EXPECTED_PYTORCH_MODEL_SHA256})."
            )
        tmp_bin.replace(pytorch_bin_path)
        print(f"Successfully downloaded and verified pytorch_model.bin ({dl_size:,} bytes, SHA-256: {dl_sha})")

    pytorch_sha256 = compute_file_sha256(pytorch_bin_path)
    pytorch_size = pytorch_bin_path.stat().st_size

    # 3. Idempotently check or export model.onnx from pinned pytorch_model.bin
    onnx_path = target_dir / "model.onnx"
    config_path = target_dir / "config.json"
    existing_cfg: Dict[str, Any] = {}
    if config_path.is_file():
        try:
            existing_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            existing_cfg = {}

    need_export = True
    inspection_info: Dict[str, Any] = {}
    if onnx_path.is_file() and not force:
        try:
            inspection_info = inspect_onnx_artifact(onnx_path)
            current_onnx_sha = compute_file_sha256(onnx_path)
            recorded_onnx_sha = existing_cfg.get("onnx_sha256")
            if recorded_onnx_sha and current_onnx_sha == recorded_onnx_sha:
                print(
                    f"Verified existing model.onnx ({onnx_path.stat().st_size:,} bytes, SHA-256: {current_onnx_sha}). Skipping re-export."
                )
                need_export = False
        except Exception as exc:
            raise RuntimeError(f"Existing model.onnx in {target_dir} is invalid or corrupt: {exc}") from exc

    exported_at = existing_cfg.get("exported_at") or datetime.now(timezone.utc).isoformat()
    if need_export:
        print(f"Exporting pinned {pytorch_bin_path.name} (SwinForImageClassification) -> {onnx_path.name} ...")
        inspection_info = export_swin_pytorch_to_onnx(pytorch_bin_path, onnx_path)
        exported_at = datetime.now(timezone.utc).isoformat()

    onnx_sha256 = compute_file_sha256(onnx_path)
    onnx_size = onnx_path.stat().st_size

    # 4. Record full integrity and provenance metadata in config.json and artifact_integrity.json
    existing_cfg.update(
        {
            "architectures": [OFFICIAL_MODEL_ARCHITECTURE],
            "model_type": "swin",
            "image_size": 224,
            "id2label": {"0": "artificial", "1": "human"},
            "label2id": {"artificial": "0", "human": "1"},
            "license": OFFICIAL_MODEL_LICENSE,
            "model_repository": model_id,
            "pinned_revision": revision,
            "onnx_is_upstream": False,
            "pytorch_model_sha256": pytorch_sha256,
            "pytorch_model_size_bytes": pytorch_size,
            "onnx_sha256": onnx_sha256,
            "onnx_size_bytes": onnx_size,
            "exported_at": exported_at,
            "conversion_info": (
                f"upstream PyTorch weights (pytorch_model.bin, {OFFICIAL_MODEL_ARCHITECTURE}) -> "
                "MediaTrace local ONNX export (opset 17) -> ONNX Runtime inference"
            ),
            "source": f"https://huggingface.co/{model_id}/tree/{revision}",
        }
    )
    config_path.write_text(json.dumps(existing_cfg, indent=2), encoding="utf-8")

    integrity_manifest = {
        "model_repository": model_id,
        "pinned_revision": revision,
        "architecture": OFFICIAL_MODEL_ARCHITECTURE,
        "license": OFFICIAL_MODEL_LICENSE,
        "onnx_is_upstream": False,
        "pytorch_model_bin": {
            "filename": "pytorch_model.bin",
            "size_bytes": pytorch_size,
            "sha256": pytorch_sha256,
            "is_upstream": True,
        },
        "model_onnx": {
            "filename": "model.onnx",
            "size_bytes": onnx_size,
            "sha256": onnx_sha256,
            "is_upstream": False,
            "derivation": "Locally exported from pinned upstream pytorch_model.bin (SwinForImageClassification)",
            "exported_at": exported_at,
            "inspection": inspection_info,
        },
    }
    (target_dir / "artifact_integrity.json").write_text(
        json.dumps(integrity_manifest, indent=2), encoding="utf-8"
    )
    return integrity_manifest


def verify_real_onnx_inference(model_id: str = OFFICIAL_MODEL_ID) -> Dict[str, Any]:
    """Perform REAL ONNX Runtime inference on a deterministic synthetic RGB image without mocking."""
    target_dir = _resolve_safe_model_directory(model_id)
    onnx_path = target_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"model.onnx not found in {target_dir}")

    onnx_sha256 = compute_file_sha256(onnx_path)
    inspection = inspect_onnx_artifact(onnx_path)

    # Create a deterministic synthetic RGB image in memory (320x240)
    img = Image.new("RGB", (320, 240), color=(48, 96, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 24, 296, 216], outline=(220, 180, 70), width=3)
    draw.line([24, 24, 296, 216], fill=(200, 90, 90), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_image_bytes = buf.getvalue()

    # Exact preprocessing pipeline (RGB, 224x224 Bicubic, [0,1], mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], NCHW float32)
    input_tensor = _preprocess_image_in_memory(raw_image_bytes)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    raw_outputs = sess.run(None, {input_name: input_tensor})
    logits = np.asarray(raw_outputs[0], dtype=np.float32)

    # Softmax + argmax class selection (0 = artificial, 1 = human)
    shifted = logits[0] - np.max(logits[0])
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    pred_idx = int(np.argmax(logits[0]))
    id2label = {0: "artificial", 1: "human"}
    selected_class = id2label[pred_idx]

    report = {
        "model_path": str(onnx_path),
        "model_sha256": onnx_sha256,
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "output_shape": list(logits.shape),
        "raw_logits": [round(float(x), 6) for x in logits[0].tolist()],
        "softmax_scores": {
            "artificial": round(float(probs[0]), 6),
            "human": round(float(probs[1]), 6),
        },
        "selected_class_index": pred_idx,
        "selected_class": selected_class,
        "onnx_inspection": inspection,
    }

    print("=== REAL ONNX RUNTIME INFERENCE VERIFICATION ===")
    print(f"model path        : {report['model_path']}")
    print(f"model SHA-256     : {report['model_sha256']}")
    print(f"input shape       : {report['input_shape']} ({report['input_dtype']})")
    print(f"output shape      : {report['output_shape']}")
    print(f"raw output values : logits={report['raw_logits']} | scores={report['softmax_scores']}")
    print(f"selected class    : {report['selected_class']} (index {report['selected_class_index']})")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provision and verify Step 14.1.1.2 pinned AI Image Detector in storage/models/"
    )
    parser.add_argument("--model-id", default=settings.ai_image_detector_model_id or OFFICIAL_MODEL_ID)
    parser.add_argument("--revision", default=settings.ai_image_detector_model_version or OFFICIAL_PINNED_REVISION)
    parser.add_argument("--force", action="store_true", help="Force re-download and re-export")
    parser.add_argument("--verify", action="store_true", help="Run real ONNX Runtime verification after setup")
    args = parser.parse_args()

    manifest = download_and_export_pinned_model(args.model_id, args.revision, force=args.force)
    print(f"Model artifacts provisioned in storage/models/: {json.dumps(manifest['model_onnx'], indent=2)}")
    if args.verify:
        verify_real_onnx_inference(args.model_id)


