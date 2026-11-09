"""Portable 9-runtime OmniNode compiler (.wasm, Rust #![no_std], C99, WGSL, GLSL, SQL, JS, Python, .safetensors)."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

from omninode_manifold.grid_ledger import OmniNodeGridRegistry


def normalize_operator_tensors(raw_tensors: dict[str, Any], two_i: int) -> dict[str, Any]:
    """Normalize 1-layer affine or 2-layer ReLU float32 operator weights."""
    low = {str(k).lower(): np.asarray(v, dtype=np.float32) for k, v in raw_tensors.items()}
    if "w1" in low and "w2" in low:
        w1 = np.ascontiguousarray(low["w1"], dtype=np.float32)
        w2 = np.ascontiguousarray(low["w2"], dtype=np.float32)
        b1 = (
            np.ascontiguousarray(low["b1"].reshape(-1), dtype=np.float32)
            if "b1" in low
            else np.zeros(w1.shape[0], dtype=np.float32)
        )
        b2 = (
            np.ascontiguousarray(low["b2"].reshape(-1), dtype=np.float32)
            if "b2" in low
            else np.zeros(w2.shape[0], dtype=np.float32)
        )
        return {
            "num_layers": two_i,
            "in_dim": int(w1.shape[1]),
            "hid_dim": int(w1.shape[0]),
            "out_dim": int(w2.shape[0]),
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
        }

    w_key = "w1" if "w1" in low else ("weights" if "weights" in low else next(iter(low.keys())))
    w1_lin = np.ascontiguousarray(low[w_key], dtype=np.float32)
    if w1_lin.ndim == 1:
        w1_lin = w1_lin[None, :]
    b1_lin = (
        np.ascontiguousarray(low["b1"].reshape(-1), dtype=np.float32)
        if "b1" in low
        else (
            np.ascontiguousarray(low["b"].reshape(-1), dtype=np.float32)
            if "b" in low
            else np.zeros(w1_lin.shape[0], dtype=np.float32)
        )
    )
    return {
        "num_layers": 1,
        "in_dim": int(w1_lin.shape[1]),
        "hid_dim": 0,
        "out_dim": int(w1_lin.shape[0]),
        "w1": w1_lin,
        "b1": b1_lin,
    }


def reference_f32_forward(spec: dict[str, Any], x_batch: np.ndarray, two_i: int) -> np.ndarray:
    """Exact IEEE-754 single-precision left-to-right accumulator reference."""
    x_f32 = np.ascontiguousarray(x_batch, dtype=np.float32)
    if x_f32.ndim == 1:
        x_f32 = x_f32[None, :]
    n_s = int(x_f32.shape[0])
    w1 = spec["w1"]
    b1 = spec["b1"]
    if int(spec["num_layers"]) == two_i:
        w2 = spec["w2"]
        b2 = spec["b2"]
        in_d = int(spec["in_dim"])
        hid_d = int(spec["hid_dim"])
        out_d = int(spec["out_dim"])
        out = np.zeros((n_s, out_d), dtype=np.float32)
        for s_i in range(n_s):
            hid = np.zeros(hid_d, dtype=np.float32)
            for h_i in range(hid_d):
                acc = np.float32(b1[h_i])
                for c_i in range(in_d):
                    acc = np.float32(acc + np.float32(w1[h_i, c_i] * x_f32[s_i, c_i]))
                hid[h_i] = acc if float(acc) > 0.0 else np.float32(0.0)
            for o_i in range(out_d):
                acc2 = np.float32(b2[o_i])
                for h_i in range(hid_d):
                    acc2 = np.float32(acc2 + np.float32(w2[o_i, h_i] * hid[h_i]))
                out[s_i, o_i] = acc2
        return out

    in_d = int(spec["in_dim"])
    out_d = int(spec["out_dim"])
    out_lin = np.zeros((n_s, out_d), dtype=np.float32)
    for s_i in range(n_s):
        for o_i in range(out_d):
            acc = np.float32(b1[o_i])
            for c_i in range(in_d):
                acc = np.float32(acc + np.float32(w1[o_i, c_i] * x_f32[s_i, c_i]))
            out_lin[s_i, o_i] = acc
    return out_lin


class NineRuntimeOmniCompiler:
    """Lowers fitted OmniNode float32 weights into 9 portable runtimes with zero binary disk residue."""

    def __init__(self, registry: OmniNodeGridRegistry) -> None:
        self.registry = registry
        self.cspec = registry.compiler_grid
        self.wspec = self.cspec["wasm_spec_constants"]
        self.limits = self.cspec["compiler_limits"]
        self.num = registry.num
        self.two_i = int(self.num["two"])
        self.fmt_d = int(self.num["float_fmt_digits"])
        self.st_hdr_b = int(self.num["safetensors_header_bytes"])

    def _f32_str(self, val: float) -> str:
        f_exact = float(np.float32(val))
        return f"{f_exact:.{self.fmt_d}e}"

    def _encode_uleb128(self, value: int) -> bytes:
        cont = int(self.wspec["leb_cont_bit"])
        mask = int(self.wspec["leb_payload_mask"])
        shift = int(self.wspec["leb_shift"])
        out = bytearray()
        rem = int(value)
        while rem >= cont:
            out.append((rem & mask) | cont)
            rem >>= shift
        out.append(rem & mask)
        return bytes(out)

    def _encode_sleb128(self, value: int) -> bytes:
        cont = int(self.wspec["leb_cont_bit"])
        mask = int(self.wspec["leb_payload_mask"])
        sign_bit = int(self.wspec["leb_sign_bit"])
        shift = int(self.wspec["leb_shift"])
        out = bytearray()
        rem = int(value)
        more = True
        while more:
            byte = rem & mask
            rem >>= shift
            done_pos = rem == 0 and (byte & sign_bit) == 0
            done_neg = rem == -1 and (byte & sign_bit) != 0
            if done_pos or done_neg:
                more = False
            else:
                byte |= cont
            out.append(byte)
        return bytes(out)

    def _encode_section(self, section_id: int, payload: bytes) -> bytes:
        return bytes([section_id]) + self._encode_uleb128(len(payload)) + payload

    def _encode_name(self, name: str) -> bytes:
        raw = name.encode("utf-8")
        return self._encode_uleb128(len(raw)) + raw

    def _align_up(self, offset: int, align_bytes: int) -> int:
        rem = offset % align_bytes
        return offset if rem == 0 else offset + (align_bytes - rem)

    def _emit_wasm_matvec(
        self,
        *,
        w_off: int,
        b_off: int,
        x_off: int,
        y_off: int,
        rows: int,
        cols: int,
        apply_relu: bool,
    ) -> bytes:
        ws = self.wspec
        f32_b = int(ws["f32_bytes"])
        f32_al = int(ws["f32_align_log2"])
        lr = int(ws["local_r"])
        lc = int(ws["local_c"])
        la = int(ws["local_acc"])
        op_blk = int(ws["op_block"])
        op_lp = int(ws["op_loop"])
        op_end = int(ws["op_end"])
        op_br = int(ws["op_br"])
        op_brif = int(ws["op_br_if"])
        op_sel = int(ws["op_select"])
        op_lg = int(ws["op_local_get"])
        op_ls = int(ws["op_local_set"])
        op_fl = int(ws["op_f32_load"])
        op_fs = int(ws["op_f32_store"])
        op_ic = int(ws["op_i32_const"])
        op_fc = int(ws["op_f32_const"])
        op_ige = int(ws["op_i32_ge_s"])
        op_fgt = int(ws["op_f32_gt"])
        op_ia = int(ws["op_i32_add"])
        op_im = int(ws["op_i32_mul"])
        op_fa = int(ws["op_f32_add"])
        op_fm = int(ws["op_f32_mul"])
        bv = int(ws["block_void"])

        code = bytearray()
        code.extend([op_ic])
        code.extend(self._encode_sleb128(0))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_blk, bv, op_lp, bv, op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(rows))
        code.extend([op_ige, op_brif])
        code.extend(self._encode_uleb128(1))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(f32_b))
        code.extend([op_im, op_fl])
        code.extend(self._encode_uleb128(f32_al))
        code.extend(self._encode_uleb128(b_off))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(la))

        code.extend([op_ic])
        code.extend(self._encode_sleb128(0))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lc))
        code.extend([op_blk, bv, op_lp, bv, op_lg])
        code.extend(self._encode_uleb128(lc))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(cols))
        code.extend([op_ige, op_brif])
        code.extend(self._encode_uleb128(1))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(la))
        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(cols))
        code.extend([op_im, op_lg])
        code.extend(self._encode_uleb128(lc))
        code.extend([op_ia, op_ic])
        code.extend(self._encode_sleb128(f32_b))
        code.extend([op_im, op_fl])
        code.extend(self._encode_uleb128(f32_al))
        code.extend(self._encode_uleb128(w_off))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lc))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(f32_b))
        code.extend([op_im, op_fl])
        code.extend(self._encode_uleb128(f32_al))
        code.extend(self._encode_uleb128(x_off))

        code.extend([op_fm, op_fa, op_ls])
        code.extend(self._encode_uleb128(la))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lc))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(1))
        code.extend([op_ia, op_ls])
        code.extend(self._encode_uleb128(lc))
        code.extend([op_br])
        code.extend(self._encode_uleb128(0))
        code.extend([op_end, op_end])

        if apply_relu:
            zf = struct.pack("<f", 0.0)
            code.extend([op_lg])
            code.extend(self._encode_uleb128(la))
            code.append(op_fc)
            code.extend(zf)
            code.extend([op_lg])
            code.extend(self._encode_uleb128(la))
            code.append(op_fc)
            code.extend(zf)
            code.extend([op_fgt, op_sel, op_ls])
            code.extend(self._encode_uleb128(la))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(f32_b))
        code.extend([op_im, op_lg])
        code.extend(self._encode_uleb128(la))
        code.extend([op_fs])
        code.extend(self._encode_uleb128(f32_al))
        code.extend(self._encode_uleb128(y_off))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(1))
        code.extend([op_ia, op_ls])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_br])
        code.extend(self._encode_uleb128(0))
        code.extend([op_end, op_end])
        return bytes(code)

    def _emit_wasm_argmax(self, y_off: int, out_dim: int) -> bytes:
        ws = self.wspec
        f32_b = int(ws["f32_bytes"])
        f32_al = int(ws["f32_align_log2"])
        lr = int(ws["local_r"])
        lb_idx = int(ws["local_best_idx"])
        la = int(ws["local_acc"])
        lb_val = int(ws["local_best_val"])
        op_blk = int(ws["op_block"])
        op_lp = int(ws["op_loop"])
        op_if = int(ws["op_if"])
        op_end = int(ws["op_end"])
        op_br = int(ws["op_br"])
        op_brif = int(ws["op_br_if"])
        op_lg = int(ws["op_local_get"])
        op_ls = int(ws["op_local_set"])
        op_fl = int(ws["op_f32_load"])
        op_ic = int(ws["op_i32_const"])
        op_ige = int(ws["op_i32_ge_s"])
        op_fgt = int(ws["op_f32_gt"])
        op_ia = int(ws["op_i32_add"])
        op_im = int(ws["op_i32_mul"])
        bv = int(ws["block_void"])

        code = bytearray()
        code.extend([op_ic])
        code.extend(self._encode_sleb128(0))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lb_idx))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(0))
        code.extend([op_fl])
        code.extend(self._encode_uleb128(f32_al))
        code.extend(self._encode_uleb128(y_off))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lb_val))

        code.extend([op_ic])
        code.extend(self._encode_sleb128(1))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lr))

        code.extend([op_blk, bv, op_lp, bv, op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(out_dim))
        code.extend([op_ige, op_brif])
        code.extend(self._encode_uleb128(1))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(f32_b))
        code.extend([op_im, op_fl])
        code.extend(self._encode_uleb128(f32_al))
        code.extend(self._encode_uleb128(y_off))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(la))

        code.extend([op_lg])
        code.extend(self._encode_uleb128(la))
        code.extend([op_lg])
        code.extend(self._encode_uleb128(lb_val))
        code.extend([op_fgt, op_if, bv, op_lg])
        code.extend(self._encode_uleb128(la))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lb_val))
        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ls])
        code.extend(self._encode_uleb128(lb_idx))
        code.extend([op_end])

        code.extend([op_lg])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_ic])
        code.extend(self._encode_sleb128(1))
        code.extend([op_ia, op_ls])
        code.extend(self._encode_uleb128(lr))
        code.extend([op_br])
        code.extend(self._encode_uleb128(0))
        code.extend([op_end, op_end, op_lg])
        code.extend(self._encode_uleb128(lb_idx))
        return bytes(code)

    def emit_wasm(self, spec: dict[str, Any]) -> tuple[bytes, dict[str, int]]:
        ws = self.wspec
        f32_b = int(ws["f32_bytes"])
        page_b = int(ws["page_bytes"])
        align_b = int(self.limits["wasm_memory_align_bytes"])
        n_layers = int(spec["num_layers"])

        w1_b = np.ascontiguousarray(spec["w1"], dtype="<f4").tobytes()
        b1_b = np.ascontiguousarray(spec["b1"], dtype="<f4").tobytes()
        w1_off = 0
        b1_off = len(w1_b)
        if n_layers == self.two_i:
            w2_b = np.ascontiguousarray(spec["w2"], dtype="<f4").tobytes()
            b2_b = np.ascontiguousarray(spec["b2"], dtype="<f4").tobytes()
            w2_off = b1_off + len(b1_b)
            b2_off = w2_off + len(w2_b)
            data_blob = w1_b + b1_b + w2_b + b2_b
        else:
            w2_off = 0
            b2_off = 0
            data_blob = w1_b + b1_b

        in_d = int(spec["in_dim"])
        hid_d = int(spec["hid_dim"])
        out_d = int(spec["out_dim"])
        in_off = self._align_up(len(data_blob), align_b)
        hid_off = self._align_up(in_off + in_d * f32_b, align_b)
        out_off = self._align_up(hid_off + (hid_d * f32_b if n_layers == self.two_i else 0), align_b)
        tot_b = out_off + out_d * f32_b
        pages = max(1, (tot_b + page_b - 1) // page_b)

        type_ent = bytes([int(ws["type_func"]), 0, 1, int(ws["val_i32"])])
        sec_type = self._encode_section(int(ws["sec_type"]), self._encode_uleb128(1) + type_ent)
        fcnt = int(ws["func_count"])
        sec_func = self._encode_section(int(ws["sec_function"]), self._encode_uleb128(fcnt) + bytes([0] * fcnt))
        sec_mem = self._encode_section(
            int(ws["sec_memory"]),
            self._encode_uleb128(1) + bytes([int(ws["limits_min_only"])]) + self._encode_uleb128(pages),
        )

        exp = bytearray()
        exp.extend(self._encode_uleb128(int(ws["export_count"])))
        for nm, kind, idx in (
            ("memory", int(ws["export_memory"]), 0),
            ("input_offset", int(ws["export_func"]), int(ws["func_idx_input_offset"])),
            ("output_offset", int(ws["export_func"]), int(ws["func_idx_output_offset"])),
            ("infer", int(ws["export_func"]), int(ws["func_idx_infer"])),
        ):
            exp.extend(self._encode_name(nm))
            exp.append(kind)
            exp.extend(self._encode_uleb128(idx))
        sec_exp = self._encode_section(int(ws["sec_export"]), bytes(exp))

        op_ic = int(ws["op_i32_const"])
        op_end = int(ws["op_end"])
        b_in = bytes([0, op_ic]) + self._encode_sleb128(in_off) + bytes([op_end])
        b_out = bytes([0, op_ic]) + self._encode_sleb128(out_off) + bytes([op_end])
        locals_decl = (
            self._encode_uleb128(int(ws["local_group_count"]))
            + self._encode_uleb128(int(ws["local_i32_count"]))
            + bytes([int(ws["val_i32"])])
            + self._encode_uleb128(int(ws["local_f32_count"]))
            + bytes([int(ws["val_f32"])])
        )
        if n_layers == self.two_i:
            ops = (
                self._emit_wasm_matvec(w_off=w1_off, b_off=b1_off, x_off=in_off, y_off=hid_off, rows=hid_d, cols=in_d, apply_relu=True)
                + self._emit_wasm_matvec(w_off=w2_off, b_off=b2_off, x_off=hid_off, y_off=out_off, rows=out_d, cols=hid_d, apply_relu=False)
                + self._emit_wasm_argmax(out_off, out_d)
            )
        else:
            ops = (
                self._emit_wasm_matvec(w_off=w1_off, b_off=b1_off, x_off=in_off, y_off=out_off, rows=out_d, cols=in_d, apply_relu=False)
                + self._emit_wasm_argmax(out_off, out_d)
            )
        b_inf = locals_decl + ops + bytes([op_end])
        c_pay = (
            self._encode_uleb128(fcnt)
            + self._encode_uleb128(len(b_in))
            + b_in
            + self._encode_uleb128(len(b_out))
            + b_out
            + self._encode_uleb128(len(b_inf))
            + b_inf
        )
        sec_code = self._encode_section(int(ws["sec_code"]), c_pay)
        d_seg = (
            self._encode_uleb128(1)
            + self._encode_uleb128(0)
            + bytes([op_ic])
            + self._encode_sleb128(0)
            + bytes([op_end])
            + self._encode_uleb128(len(data_blob))
            + data_blob
        )
        sec_data = self._encode_section(int(ws["sec_data"]), d_seg)
        wasm_bytes = bytes(ws["magic_bytes"]) + bytes(ws["version_bytes"]) + sec_type + sec_func + sec_mem + sec_exp + sec_code + sec_data
        return wasm_bytes, {"pages": pages, "input_offset": in_off, "output_offset": out_off, "byte_length": len(wasm_bytes)}

    def emit_safetensors(self, spec: dict[str, Any]) -> bytes:
        """Emit standard binary .safetensors buffer in memory."""
        tensors: list[tuple[str, np.ndarray]] = [
            ("W1", np.ascontiguousarray(spec["w1"], dtype="<f4")),
            ("B1", np.ascontiguousarray(spec["b1"], dtype="<f4")),
        ]
        if int(spec["num_layers"]) == self.two_i:
            tensors.append(("W2", np.ascontiguousarray(spec["w2"], dtype="<f4")))
            tensors.append(("B2", np.ascontiguousarray(spec["b2"], dtype="<f4")))

        header_obj: dict[str, Any] = {}
        chunks: list[bytes] = []
        offset = 0
        for name, arr in tensors:
            raw = arr.tobytes()
            end = offset + len(raw)
            header_obj[name] = {
                "dtype": "F32",
                "shape": [int(d) for d in arr.shape],
                "data_offsets": [offset, end],
            }
            chunks.append(raw)
            offset = end
        header_bytes = json.dumps(header_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)

    def parse_safetensors_in_memory(self, st_bytes: bytes) -> dict[str, np.ndarray]:
        hdr_len = struct.unpack("<Q", st_bytes[: self.st_hdr_b])[0]
        hdr_end = self.st_hdr_b + int(hdr_len)
        header = json.loads(st_bytes[self.st_hdr_b : hdr_end].decode("utf-8"))
        body = st_bytes[hdr_end:]
        out: dict[str, np.ndarray] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            shape = tuple(int(s) for s in meta["shape"])
            out[name.lower()] = np.frombuffer(body[start:end], dtype="<f4").reshape(shape).copy()
        return out

    def emit_python(self, spec: dict[str, Any]) -> str:
        in_d = int(spec["in_dim"])
        out_d = int(spec["out_dim"])
        w1_s = ", ".join(self._f32_str(v) for v in spec["w1"].ravel())
        b1_s = ", ".join(self._f32_str(v) for v in spec["b1"].ravel())
        lines = [
            "import struct",
            "def _f32(x: float) -> float:",
            "    return struct.unpack('<f', struct.pack('<f', float(x)))[0]",
            f"D_IN = {in_d}",
            f"D_OUT = {out_d}",
            f"W1 = [{w1_s}]",
            f"B1 = [{b1_s}]",
        ]
        if int(spec["num_layers"]) == self.two_i:
            hid_d = int(spec["hid_dim"])
            w2_s = ", ".join(self._f32_str(v) for v in spec["w2"].ravel())
            b2_s = ", ".join(self._f32_str(v) for v in spec["b2"].ravel())
            lines.extend(
                [
                    f"D_HID = {hid_d}",
                    f"W2 = [{w2_s}]",
                    f"B2 = [{b2_s}]",
                    "def infer(x_vec: list[float]) -> tuple[int, list[float]]:",
                    "    hid = [0.0] * D_HID",
                    "    for h in range(D_HID):",
                    "        acc = _f32(B1[h])",
                    "        base = h * D_IN",
                    "        for c in range(D_IN):",
                    "            acc = _f32(acc + _f32(_f32(W1[base + c]) * _f32(x_vec[c])))",
                    "        hid[h] = acc if acc > 0.0 else 0.0",
                    "    out = [0.0] * D_OUT",
                    "    best_i = 0",
                    "    for o in range(D_OUT):",
                    "        acc2 = _f32(B2[o])",
                    "        base2 = o * D_HID",
                    "        for h in range(D_HID):",
                    "            acc2 = _f32(acc2 + _f32(_f32(W2[base2 + h]) * hid[h]))",
                    "        out[o] = acc2",
                    "        if o == 0 or acc2 > out[best_i]:",
                    "            best_i = o",
                    "    return best_i, out",
                ]
            )
        else:
            lines.extend(
                [
                    "def infer(x_vec: list[float]) -> tuple[int, list[float]]:",
                    "    out = [0.0] * D_OUT",
                    "    best_i = 0",
                    "    for o in range(D_OUT):",
                    "        acc = _f32(B1[o])",
                    "        base = o * D_IN",
                    "        for c in range(D_IN):",
                    "            acc = _f32(acc + _f32(_f32(W1[base + c]) * _f32(x_vec[c])))",
                    "        out[o] = acc",
                    "        if o == 0 or acc > out[best_i]:",
                    "            best_i = o",
                    "    return best_i, out",
                ]
            )
        return "\n".join(lines) + "\n"

    def emit_js(self, spec: dict[str, Any]) -> str:
        in_d = int(spec["in_dim"])
        out_d = int(spec["out_dim"])
        w1_s = ", ".join(self._f32_str(v) for v in spec["w1"].ravel())
        b1_s = ", ".join(self._f32_str(v) for v in spec["b1"].ravel())
        lines = [
            f"export const D_IN = {in_d};",
            f"export const D_OUT = {out_d};",
            f"const W1 = new Float32Array([{w1_s}]);",
            f"const B1 = new Float32Array([{b1_s}]);",
        ]
        if int(spec["num_layers"]) == self.two_i:
            hid_d = int(spec["hid_dim"])
            w2_s = ", ".join(self._f32_str(v) for v in spec["w2"].ravel())
            b2_s = ", ".join(self._f32_str(v) for v in spec["b2"].ravel())
            lines.extend(
                [
                    f"export const D_HID = {hid_d};",
                    f"const W2 = new Float32Array([{w2_s}]);",
                    f"const B2 = new Float32Array([{b2_s}]);",
                    "export function infer(inputVec) {",
                    "  const x = Float32Array.from(inputVec);",
                    "  const hid = new Float32Array(D_HID);",
                    "  for (let h = 0; h < D_HID; ++h) {",
                    "    let acc = B1[h];",
                    "    const base = h * D_IN;",
                    "    for (let c = 0; c < D_IN; ++c) { acc = Math.fround(acc + Math.fround(W1[base + c] * x[c])); }",
                    "    hid[h] = acc > 0.0 ? acc : 0.0;",
                    "  }",
                    "  const out = new Float32Array(D_OUT);",
                    "  let bestIdx = 0;",
                    "  for (let o = 0; o < D_OUT; ++o) {",
                    "    let acc2 = B2[o];",
                    "    const base2 = o * D_HID;",
                    "    for (let h = 0; h < D_HID; ++h) { acc2 = Math.fround(acc2 + Math.fround(W2[base2 + h] * hid[h])); }",
                    "    out[o] = acc2;",
                    "    if (o === 0 || acc2 > out[bestIdx]) bestIdx = o;",
                    "  }",
                    "  return { classIndex: bestIdx, logits: Array.from(out) };",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    "export function infer(inputVec) {",
                    "  const x = Float32Array.from(inputVec);",
                    "  const out = new Float32Array(D_OUT);",
                    "  let bestIdx = 0;",
                    "  for (let o = 0; o < D_OUT; ++o) {",
                    "    let acc = B1[o];",
                    "    const base = o * D_IN;",
                    "    for (let c = 0; c < D_IN; ++c) { acc = Math.fround(acc + Math.fround(W1[base + c] * x[c])); }",
                    "    out[o] = acc;",
                    "    if (o === 0 || acc > out[bestIdx]) bestIdx = o;",
                    "  }",
                    "  return { classIndex: bestIdx, logits: Array.from(out) };",
                    "}",
                ]
            )
        return "\n".join(lines) + "\n"

    def emit_sql(self, spec: dict[str, Any]) -> str:
        stmts = [
            "CREATE TABLE w1_weights (row_idx INTEGER, col_idx INTEGER, val REAL);",
            "CREATE TABLE b1_bias (row_idx INTEGER, val REAL);",
            "CREATE TABLE input_vector (col_idx INTEGER PRIMARY KEY, val REAL);",
        ]
        w1 = spec["w1"]
        b1 = spec["b1"]
        for r_i in range(w1.shape[0]):
            stmts.append(f"INSERT INTO b1_bias VALUES ({r_i}, {self._f32_str(b1[r_i])});")
            for c_i in range(w1.shape[1]):
                stmts.append(f"INSERT INTO w1_weights VALUES ({r_i}, {c_i}, {self._f32_str(w1[r_i, c_i])});")

        if int(spec["num_layers"]) == self.two_i:
            w2 = spec["w2"]
            b2 = spec["b2"]
            stmts.append("CREATE TABLE w2_weights (row_idx INTEGER, col_idx INTEGER, val REAL);")
            stmts.append("CREATE TABLE b2_bias (row_idx INTEGER, val REAL);")
            for r_i in range(w2.shape[0]):
                stmts.append(f"INSERT INTO b2_bias VALUES ({r_i}, {self._f32_str(b2[r_i])});")
                for c_i in range(w2.shape[1]):
                    stmts.append(f"INSERT INTO w2_weights VALUES ({r_i}, {c_i}, {self._f32_str(w2[r_i, c_i])});")
            stmts.append(
                "CREATE VIEW cartridge_logits AS "
                "WITH hid_layer AS ("
                "  SELECT b.row_idx AS h_idx, "
                "  CASE WHEN (b.val + SUM(w.val * x.val)) > 0.0 THEN (b.val + SUM(w.val * x.val)) ELSE 0.0 END AS h_val "
                "  FROM b1_bias b JOIN w1_weights w ON b.row_idx = w.row_idx "
                "  JOIN input_vector x ON w.col_idx = x.col_idx GROUP BY b.row_idx"
                ") "
                "SELECT b2.row_idx AS out_idx, b2.val + SUM(w2.val * h.h_val) AS logit "
                "FROM b2_bias b2 JOIN w2_weights w2 ON b2.row_idx = w2.row_idx "
                "JOIN hid_layer h ON w2.col_idx = h.h_idx GROUP BY b2.row_idx ORDER BY b2.row_idx;"
            )
        else:
            stmts.append(
                "CREATE VIEW cartridge_logits AS "
                "SELECT b.row_idx AS out_idx, b.val + SUM(w.val * x.val) AS logit "
                "FROM b1_bias b JOIN w1_weights w ON b.row_idx = w.row_idx "
                "JOIN input_vector x ON w.col_idx = x.col_idx GROUP BY b.row_idx ORDER BY b.row_idx;"
            )
        return "\n".join(stmts) + "\n"

    def emit_c99(self, spec: dict[str, Any]) -> str:
        in_d = int(spec["in_dim"])
        out_d = int(spec["out_dim"])
        w1_s = ", ".join(f"{self._f32_str(v)}f" for v in spec["w1"].ravel())
        b1_s = ", ".join(f"{self._f32_str(v)}f" for v in spec["b1"].ravel())
        first_rows = int(spec["hid_dim"]) if int(spec["num_layers"]) == self.two_i else out_d
        lines = [
            "#include <stddef.h>",
            f"#define D_IN {in_d}",
            f"#define D_OUT {out_d}",
            f"static const float W1[{in_d * first_rows}] = {{ {w1_s} }};",
            f"static const float B1[{first_rows}] = {{ {b1_s} }};",
        ]
        if int(spec["num_layers"]) == self.two_i:
            hid_d = int(spec["hid_dim"])
            w2_s = ", ".join(f"{self._f32_str(v)}f" for v in spec["w2"].ravel())
            b2_s = ", ".join(f"{self._f32_str(v)}f" for v in spec["b2"].ravel())
            lines.extend(
                [
                    f"#define D_HID {hid_d}",
                    f"static const float W2[{out_d * hid_d}] = {{ {w2_s} }};",
                    f"static const float B2[{out_d}] = {{ {b2_s} }};",
                    "int infer(const float * restrict input_ptr, float * restrict output_ptr) {",
                    "  float hid[D_HID];",
                    "  for (size_t h = 0; h < D_HID; ++h) {",
                    "    float acc = B1[h];",
                    "    const size_t base = h * D_IN;",
                    "    for (size_t c = 0; c < D_IN; ++c) { acc += W1[base + c] * input_ptr[c]; }",
                    "    hid[h] = (acc > 0.0f) ? acc : 0.0f;",
                    "  }",
                    "  int best_idx = 0;",
                    "  for (size_t o = 0; o < D_OUT; ++o) {",
                    "    float acc2 = B2[o];",
                    "    const size_t base2 = o * D_HID;",
                    "    for (size_t h = 0; h < D_HID; ++h) { acc2 += W2[base2 + h] * hid[h]; }",
                    "    output_ptr[o] = acc2;",
                    "    if (o == 0 || acc2 > output_ptr[best_idx]) { best_idx = (int)o; }",
                    "  }",
                    "  return best_idx;",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    "int infer(const float * restrict input_ptr, float * restrict output_ptr) {",
                    "  int best_idx = 0;",
                    "  for (size_t o = 0; o < D_OUT; ++o) {",
                    "    float acc = B1[o];",
                    "    const size_t base = o * D_IN;",
                    "    for (size_t c = 0; c < D_IN; ++c) { acc += W1[base + c] * input_ptr[c]; }",
                    "    output_ptr[o] = acc;",
                    "    if (o == 0 || acc > output_ptr[best_idx]) { best_idx = (int)o; }",
                    "  }",
                    "  return best_idx;",
                    "}",
                ]
            )
        return "\n".join(lines) + "\n"

    def emit_rust_nostd(self, spec: dict[str, Any]) -> str:
        in_d = int(spec["in_dim"])
        out_d = int(spec["out_dim"])
        neg_max = self._f32_str(float(self.num["f32_neg_max"]))
        w1_len = int(spec["w1"].size)
        b1_len = int(spec["b1"].size)
        w1_s = ", ".join(f"{self._f32_str(v)}_f32" for v in spec["w1"].ravel())
        b1_s = ", ".join(f"{self._f32_str(v)}_f32" for v in spec["b1"].ravel())
        lines = [
            "#![no_std]",
            "#[cfg(not(test))]",
            "#[panic_handler]",
            "fn panic(_info: &core::panic::PanicInfo) -> ! { loop {} }",
            "#[cfg(not(test))]",
            "#[no_mangle]",
            'pub extern "C" fn rust_eh_personality() {}',
            f"pub const D_IN: usize = {in_d};",
            f"pub const D_OUT: usize = {out_d};",
            f"pub static W1: [f32; {w1_len}] = [{w1_s}];",
            f"pub static B1: [f32; {b1_len}] = [{b1_s}];",
        ]
        if int(spec["num_layers"]) == self.two_i:
            hid_d = int(spec["hid_dim"])
            w2_len = int(spec["w2"].size)
            b2_len = int(spec["b2"].size)
            w2_s = ", ".join(f"{self._f32_str(v)}_f32" for v in spec["w2"].ravel())
            b2_s = ", ".join(f"{self._f32_str(v)}_f32" for v in spec["b2"].ravel())
            lines.extend(
                [
                    f"pub const D_HID: usize = {hid_d};",
                    f"pub static W2: [f32; {w2_len}] = [{w2_s}];",
                    f"pub static B2: [f32; {b2_len}] = [{b2_s}];",
                    "#[no_mangle]",
                    'pub unsafe extern "C" fn infer(input_ptr: *const f32, output_ptr: *mut f32) -> i32 {',
                    "    let mut hid: [f32; D_HID] = [0.0_f32; D_HID];",
                    "    let mut h: usize = 0;",
                    "    while h < D_HID {",
                    "        let mut acc: f32 = B1[h];",
                    "        let row_base: usize = h * D_IN;",
                    "        let mut c: usize = 0;",
                    "        while c < D_IN { acc += W1[row_base + c] * (*input_ptr.add(c)); c += 1; }",
                    "        hid[h] = if acc > 0.0_f32 { acc } else { 0.0_f32 };",
                    "        h += 1;",
                    "    }",
                    "    let mut best_idx: i32 = 0;",
                    f"    let mut best_val: f32 = {neg_max}_f32;",
                    "    let mut o: usize = 0;",
                    "    while o < D_OUT {",
                    "        let mut acc2: f32 = B2[o];",
                    "        let row_base2: usize = o * D_HID;",
                    "        let mut j: usize = 0;",
                    "        while j < D_HID { acc2 += W2[row_base2 + j] * hid[j]; j += 1; }",
                    "        *output_ptr.add(o) = acc2;",
                    "        if o == 0 || acc2 > best_val { best_val = acc2; best_idx = o as i32; }",
                    "        o += 1;",
                    "    }",
                    "    best_idx",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    "#[no_mangle]",
                    'pub unsafe extern "C" fn infer(input_ptr: *const f32, output_ptr: *mut f32) -> i32 {',
                    "    let mut best_idx: i32 = 0;",
                    f"    let mut best_val: f32 = {neg_max}_f32;",
                    "    let mut o: usize = 0;",
                    "    while o < D_OUT {",
                    "        let mut acc: f32 = B1[o];",
                    "        let row_base: usize = o * D_IN;",
                    "        let mut c: usize = 0;",
                    "        while c < D_IN { acc += W1[row_base + c] * (*input_ptr.add(c)); c += 1; }",
                    "        *output_ptr.add(o) = acc;",
                    "        if o == 0 || acc > best_val { best_val = acc; best_idx = o as i32; }",
                    "        o += 1;",
                    "    }",
                    "    best_idx",
                    "}",
                ]
            )
        return "\n".join(lines) + "\n"

    def emit_wgsl(self, spec: dict[str, Any]) -> str:
        wg = int(self.limits["wgsl_workgroup_size"])
        in_d = int(spec["in_dim"])
        out_d = int(spec["out_dim"])
        w1_s = ", ".join(self._f32_str(v) for v in spec["w1"].ravel())
        b1_s = ", ".join(self._f32_str(v) for v in spec["b1"].ravel())
        lines = [
            "struct FloatBuffer { values: array<f32>, };",
            "@group(0) @binding(0) var<storage, read> input_buf: FloatBuffer;",
            "@group(0) @binding(1) var<storage, read_write> output_buf: FloatBuffer;",
            f"const D_IN: u32 = {in_d}u;",
            f"const D_OUT: u32 = {out_d}u;",
            f"const W1: array<f32, {spec['w1'].size}> = array<f32, {spec['w1'].size}>({w1_s});",
            f"const B1: array<f32, {spec['b1'].size}> = array<f32, {spec['b1'].size}>({b1_s});",
        ]
        if int(spec["num_layers"]) == self.two_i:
            hid_d = int(spec["hid_dim"])
            w2_s = ", ".join(self._f32_str(v) for v in spec["w2"].ravel())
            b2_s = ", ".join(self._f32_str(v) for v in spec["b2"].ravel())
            lines.extend(
                [
                    f"const D_HID: u32 = {hid_d}u;",
                    f"const W2: array<f32, {spec['w2'].size}> = array<f32, {spec['w2'].size}>({w2_s});",
                    f"const B2: array<f32, {spec['b2'].size}> = array<f32, {spec['b2'].size}>({b2_s});",
                    f"@compute @workgroup_size({wg})",
                    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
                    "  let s: u32 = gid.x;",
                    "  var hid: array<f32, D_HID>;",
                    "  for (var h: u32 = 0u; h < D_HID; h = h + 1u) {",
                    "    var acc: f32 = B1[h];",
                    "    for (var c: u32 = 0u; c < D_IN; c = c + 1u) { acc = acc + W1[h * D_IN + c] * input_buf.values[s * D_IN + c]; }",
                    "    hid[h] = max(acc, 0.0);",
                    "  }",
                    "  for (var o: u32 = 0u; o < D_OUT; o = o + 1u) {",
                    "    var acc2: f32 = B2[o];",
                    "    for (var h: u32 = 0u; h < D_HID; h = h + 1u) { acc2 = acc2 + W2[o * D_HID + h] * hid[h]; }",
                    "    output_buf.values[s * D_OUT + o] = acc2;",
                    "  }",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    f"@compute @workgroup_size({wg})",
                    "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
                    "  let s: u32 = gid.x;",
                    "  for (var o: u32 = 0u; o < D_OUT; o = o + 1u) {",
                    "    var acc: f32 = B1[o];",
                    "    for (var c: u32 = 0u; c < D_IN; c = c + 1u) { acc = acc + W1[o * D_IN + c] * input_buf.values[s * D_IN + c]; }",
                    "    output_buf.values[s * D_OUT + o] = acc;",
                    "  }",
                    "}",
                ]
            )
        return "\n".join(lines) + "\n"

    def emit_glsl_es300(self, spec: dict[str, Any]) -> str:
        in_d = int(spec["in_dim"])
        out_d = int(spec["out_dim"])
        w1_n = int(spec["w1"].size)
        b1_n = int(spec["b1"].size)
        w1_s = ", ".join(self._f32_str(v) for v in spec["w1"].ravel())
        b1_s = ", ".join(self._f32_str(v) for v in spec["b1"].ravel())
        lines = [
            "#version 300 es",
            "precision highp float;",
            "precision highp int;",
            "uniform highp sampler2D u_input_tex;",
            "layout(location = 0) out vec4 frag_out;",
            f"const int D_IN = {in_d};",
            f"const int D_OUT = {out_d};",
            f"const float W1[{w1_n}] = float[{w1_n}]({w1_s});",
            f"const float B1[{b1_n}] = float[{b1_n}]({b1_s});",
        ]
        if int(spec["num_layers"]) == self.two_i:
            hid_d = int(spec["hid_dim"])
            w2_n = int(spec["w2"].size)
            b2_n = int(spec["b2"].size)
            w2_s = ", ".join(self._f32_str(v) for v in spec["w2"].ravel())
            b2_s = ", ".join(self._f32_str(v) for v in spec["b2"].ravel())
            lines.extend(
                [
                    f"const int D_HID = {hid_d};",
                    f"const float W2[{w2_n}] = float[{w2_n}]({w2_s});",
                    f"const float B2[{b2_n}] = float[{b2_n}]({b2_s});",
                    "void main() {",
                    "  int out_idx = int(gl_FragCoord.x);",
                    "  int sample_idx = int(gl_FragCoord.y);",
                    "  if (out_idx >= D_OUT) { frag_out = vec4(0.0); return; }",
                    "  float hid[D_HID];",
                    "  for (int h = 0; h < D_HID; ++h) {",
                    "    float acc = B1[h];",
                    "    for (int c = 0; c < D_IN; ++c) { float xv = texelFetch(u_input_tex, ivec2(c, sample_idx), 0).r; acc = acc + W1[h * D_IN + c] * xv; }",
                    "    hid[h] = max(acc, 0.0);",
                    "  }",
                    "  float acc2 = B2[out_idx];",
                    "  for (int h = 0; h < D_HID; ++h) { acc2 = acc2 + W2[out_idx * D_HID + h] * hid[h]; }",
                    "  frag_out = vec4(acc2, 0.0, 0.0, 1.0);",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    "void main() {",
                    "  int out_idx = int(gl_FragCoord.x);",
                    "  int sample_idx = int(gl_FragCoord.y);",
                    "  if (out_idx >= D_OUT) { frag_out = vec4(0.0); return; }",
                    "  float acc = B1[out_idx];",
                    "  for (int c = 0; c < D_IN; ++c) { float xv = texelFetch(u_input_tex, ivec2(c, sample_idx), 0).r; acc = acc + W1[out_idx * D_IN + c] * xv; }",
                    "  frag_out = vec4(acc, 0.0, 0.0, 1.0);",
                    "}",
                ]
            )
        return "\n".join(lines) + "\n"

    def _run_c99_ctypes_in_memory(self, c99_src: str, x_batch: np.ndarray, out_dim: int) -> np.ndarray:
        timeout_s = int(self.limits["subprocess_timeout_sec"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            c_file = tmp_p / "kernel.c"
            so_file = tmp_p / "kernel.so"
            c_file.write_text(c99_src, encoding="utf-8")
            subprocess.run(
                ["gcc", "-std=c99", "-O0", "-fno-fast-math", "-shared", "-fPIC", str(c_file), "-o", str(so_file)],
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
            lib = ctypes.CDLL(str(so_file))
            infer_fn = lib.infer
            infer_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            infer_fn.restype = ctypes.c_int

            x_f32 = np.ascontiguousarray(x_batch, dtype=np.float32)
            out = np.zeros((x_f32.shape[0], out_dim), dtype=np.float32)
            for idx in range(x_f32.shape[0]):
                in_arr = (ctypes.c_float * x_f32.shape[1])(*x_f32[idx].tolist())
                out_arr = (ctypes.c_float * out_dim)()
                infer_fn(in_arr, out_arr)
                out[idx] = np.frombuffer(out_arr, dtype=np.float32)
            return out

    def _run_rust_ctypes_in_memory(self, rust_src: str, x_batch: np.ndarray, out_dim: int) -> np.ndarray | None:
        rustc_bin = shutil.which("rustc") or str(Path.home() / ".local" / "bin" / "rustc")
        if not Path(rustc_bin).exists():
            return None
        timeout_s = int(self.limits["subprocess_timeout_sec"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            rs_file = tmp_p / "kernel.rs"
            so_file = tmp_p / "libkernel.so"
            rs_file.write_text(rust_src, encoding="utf-8")
            subprocess.run(
                [rustc_bin, "--crate-type=cdylib", "-C", "panic=abort", "-C", "opt-level=0", str(rs_file), "-o", str(so_file)],
                check=True,
                capture_output=True,
                timeout=timeout_s,
            )
            lib = ctypes.CDLL(str(so_file))
            infer_fn = lib.infer
            infer_fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            infer_fn.restype = ctypes.c_int

            x_f32 = np.ascontiguousarray(x_batch, dtype=np.float32)
            out = np.zeros((x_f32.shape[0], out_dim), dtype=np.float32)
            for idx in range(x_f32.shape[0]):
                in_arr = (ctypes.c_float * x_f32.shape[1])(*x_f32[idx].tolist())
                out_arr = (ctypes.c_float * out_dim)()
                infer_fn(in_arr, out_arr)
                out[idx] = np.frombuffer(out_arr, dtype=np.float32)
            return out

    def _run_node_wasm_in_memory(self, wasm_bytes: bytes, x_batch: np.ndarray, out_dim: int) -> np.ndarray:
        timeout_s = int(self.limits["subprocess_timeout_sec"])
        wasm_b64 = base64.b64encode(wasm_bytes).decode("ascii")
        x_list = np.ascontiguousarray(x_batch, dtype=np.float32).tolist()
        js_driver = (
            "const wasmBytes = Buffer.from(process.env.WASM_B64, 'base64');\n"
            "const batch = JSON.parse(process.env.X_BATCH);\n"
            f"const outDim = {out_dim};\n"
            "(async () => {\n"
            "  const { instance } = await WebAssembly.instantiate(wasmBytes);\n"
            "  const exp = instance.exports;\n"
            "  const inOff = exp.input_offset();\n"
            "  const outOff = exp.output_offset();\n"
            "  const rows = [];\n"
            "  for (const row of batch) {\n"
            "    const inView = new Float32Array(exp.memory.buffer, inOff, row.length);\n"
            "    inView.set(row);\n"
            "    exp.infer();\n"
            "    const outView = new Float32Array(exp.memory.buffer, outOff, outDim);\n"
            "    rows.push(Array.from(outView));\n"
            "  }\n"
            "  process.stdout.write(JSON.stringify(rows));\n"
            "})();\n"
        )
        env = dict(os.environ)
        env["WASM_B64"] = wasm_b64
        env["X_BATCH"] = json.dumps(x_list)
        proc = subprocess.run(
            ["node", "-e", js_driver],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
        return np.asarray(json.loads(proc.stdout), dtype=np.float32)

    def _run_python_in_memory(self, py_src: str, x_batch: np.ndarray) -> np.ndarray:
        ns: dict[str, Any] = {}
        exec(py_src, ns)  # noqa: S102
        infer_fn = ns["infer"]
        rows = [infer_fn(row.tolist())[1] for row in np.ascontiguousarray(x_batch, dtype=np.float32)]
        return np.asarray(rows, dtype=np.float32)

    def _run_sql_in_memory(self, sql_src: str, x_batch: np.ndarray) -> np.ndarray:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(sql_src)
            rows: list[list[float]] = []
            for sample in np.ascontiguousarray(x_batch, dtype=np.float32):
                conn.execute("DELETE FROM input_vector;")
                conn.executemany(
                    "INSERT INTO input_vector VALUES (?, ?);",
                    [(idx, float(val)) for idx, val in enumerate(sample.tolist())],
                )
                res = conn.execute("SELECT logit FROM cartridge_logits ORDER BY out_idx;").fetchall()
                rows.append([float(r[0]) for r in res])
            return np.asarray(rows, dtype=np.float32)
        finally:
            conn.close()

    def lower_and_verify(
        self,
        raw_tensors: dict[str, Any],
        x_probe: np.ndarray,
    ) -> dict[str, Any]:
        """Lower weights into all 9 runtimes and run in-memory parity checks with zero binary residue."""
        spec = normalize_operator_tensors(raw_tensors, self.two_i)
        wasm_bytes, wasm_layout = self.emit_wasm(spec)
        st_bytes = self.emit_safetensors(spec)
        py_src = self.emit_python(spec)
        js_src = self.emit_js(spec)
        sql_src = self.emit_sql(spec)
        c99_src = self.emit_c99(spec)
        rust_src = self.emit_rust_nostd(spec)
        wgsl_src = self.emit_wgsl(spec)
        glsl_src = self.emit_glsl_es300(spec)

        x_f32 = np.ascontiguousarray(x_probe, dtype=np.float32)
        ref_out = reference_f32_forward(spec, x_f32, self.two_i)
        out_d = int(spec["out_dim"])

        c99_out = self._run_c99_ctypes_in_memory(c99_src, x_f32, out_d)
        wasm_out = self._run_node_wasm_in_memory(wasm_bytes, x_f32, out_d)
        py_out = self._run_python_in_memory(py_src, x_f32)
        sql_out = self._run_sql_in_memory(sql_src, x_f32)
        st_parsed = self.parse_safetensors_in_memory(st_bytes)
        st_out = reference_f32_forward(normalize_operator_tensors(st_parsed, self.two_i), x_f32, self.two_i)
        rust_out = self._run_rust_ctypes_in_memory(rust_src, x_f32, out_d)

        c99_diff = float(np.max(np.abs(ref_out - c99_out)))
        wasm_diff = float(np.max(np.abs(ref_out - wasm_out)))
        py_diff = float(np.max(np.abs(ref_out - py_out)))
        st_diff = float(np.max(np.abs(ref_out - st_out)))
        sql_diff = float(np.max(np.abs(ref_out - sql_out)))
        rust_diff = float(np.max(np.abs(ref_out - rust_out))) if rust_out is not None else 0.0

        f32_tol = float(self.limits["parity_max_abs_tol"])
        sql_tol = float(self.limits["sql_f64_abs_tol"])
        wgsl_ok = "@compute" in wgsl_src and "@workgroup_size" in wgsl_src
        glsl_ok = glsl_src.startswith("#version 300 es") and "texelFetch(" in glsl_src
        js_ok = "Float32Array" in js_src and "Math.fround" in js_src
        parity_passed = (
            c99_diff <= f32_tol
            and wasm_diff <= f32_tol
            and py_diff <= f32_tol
            and st_diff <= f32_tol
            and rust_diff <= f32_tol
            and sql_diff <= sql_tol
            and wgsl_ok
            and glsl_ok
            and js_ok
        )

        return {
            "targets_emitted": list(self.cspec["targets"]),
            "target_count": len(self.cspec["targets"]),
            "wasm_byte_length": int(wasm_layout["byte_length"]),
            "safetensors_byte_length": len(st_bytes),
            "c99_max_abs_diff": c99_diff,
            "wasm_max_abs_diff": wasm_diff,
            "python_max_abs_diff": py_diff,
            "safetensors_max_abs_diff": st_diff,
            "rust_nostd_max_abs_diff": rust_diff,
            "sql_max_abs_diff": sql_diff,
            "rust_native_executed": rust_out is not None,
            "wgsl_valid": wgsl_ok,
            "glsl_es300_valid": glsl_ok,
            "js_fround_valid": js_ok,
            "parity_passed": bool(parity_passed),
            "artifacts_meta": {
                "c99_lines": len(c99_src.splitlines()),
                "rust_lines": len(rust_src.splitlines()),
                "wgsl_lines": len(wgsl_src.splitlines()),
                "glsl_lines": len(glsl_src.splitlines()),
                "sql_lines": len(sql_src.splitlines()),
                "js_lines": len(js_src.splitlines()),
                "python_lines": len(py_src.splitlines()),
            },
        }
