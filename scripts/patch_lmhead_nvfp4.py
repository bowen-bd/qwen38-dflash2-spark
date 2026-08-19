#!/usr/bin/env python3
"""Dequantize an NVFP4 lm_head to dense BF16 so SGLang's DFlash2 selector can use it.

DFlash2's candidate selector reads the target's lm_head directly to get top-K
candidates per position, and refuses anything it cannot read as a plain matrix:

    dflash.py:969  RuntimeError: DFlash2 selector requires a dense FP16/BF16/FP32
                                 target lm_head.
    (is_dense_head_weight() just tests weight.dtype in {F16, BF16, F32})

RadixArk/Qwen3.8-27B-NVFP4 stores lm_head as NVFP4: weight U8 [vocab, hidden/2]
(two E2M1 nibbles per byte), weight_scale F8_E4M3 [vocab, hidden/16] (one FP8
scale per 16-element block), and a global F32 weight_scale_2. So:

    W[i,j] = e2m1(nibble) * f8e4m3(weight_scale[i, j//16]) * weight_scale_2

Nibble order was verified empirically, not assumed: element 2k is the LOW nibble
of byte k. Decoded against the official FP8 checkpoint's dense BF16 lm_head,
low-nibble-first scores cosine 0.9955 (= NVFP4's own quantization error) while
high-nibble-first scores 0.0016. Note this is the sibling of patch_lmhead.py,
which does the same job for Unsloth's FP8 head for vLLM -- same failure class,
different encoding.

Costs +1.83 GB on disk. Every other tensor is byte-identical, and unchanged
shards are hardlinked, so this adds ~3.8 GB rather than a full 21 GB copy.
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

HOME = os.environ.get("QWEN_HOME", os.path.expanduser("~/llm"))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "radixark-nvfp4")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "radixark-nvfp4-dflash2")
ROWS = 8192

DROP = ("lm_head.weight_scale", "lm_head.weight_scale_2", "lm_head.input_scale")
WK = "lm_head.weight"


def f8e4m3_table():
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * (2.0**-6)
        elif e == 0xF and m == 0x7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        out[b] = s * v
    return out


F8 = f8e4m3_table()
# E2M1: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign.
E2M1 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                 -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)


def f32_to_bf16_bits(x):
    u = x.view(np.uint32)
    rounding = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return ((u + rounding) >> 16).astype(np.uint16)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", None)
    return hdr, meta, 8 + n


def main():
    os.makedirs(DST, exist_ok=True)
    idx_src = os.path.join(SRC, "model.safetensors.index.json")
    wmap = json.load(open(idx_src))["weight_map"]
    shard = wmap[WK]
    print(f"lm_head lives in {shard}")

    src_st = os.path.join(SRC, shard)
    hdr, meta, data_start = read_header(src_st)
    assert hdr[WK]["dtype"] == "U8", hdr[WK]["dtype"]
    vocab, packed = hdr[WK]["shape"]
    hidden = packed * 2
    nblk = hdr["lm_head.weight_scale"]["shape"][1]
    bs = hidden // nblk
    print(f"vocab={vocab} hidden={hidden} block_size={bs}")

    with open(src_st, "rb") as f:
        o = hdr["lm_head.weight_scale_2"]["data_offsets"]
        f.seek(data_start + o[0])
        s2 = np.frombuffer(f.read(4), dtype=np.float32)[0]
    print(f"weight_scale_2={s2}")

    order = sorted((k for k in hdr if k not in DROP),
                   key=lambda k: hdr[k]["data_offsets"][0])
    new_hdr, cur = {}, 0
    for k in order:
        if k == WK:
            nbytes = vocab * hidden * 2
            new_hdr[k] = {"dtype": "BF16", "shape": [vocab, hidden],
                          "data_offsets": [cur, cur + nbytes]}
        else:
            lo, hi = hdr[k]["data_offsets"]
            nbytes = hi - lo
            new_hdr[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"],
                          "data_offsets": [cur, cur + nbytes]}
        cur += nbytes

    obj = dict(new_hdr)
    if meta is not None:
        obj["__metadata__"] = meta
    blob = json.dumps(obj, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)

    dst_st = os.path.join(DST, shard)
    print(f"writing {dst_st} ({cur/2**30:.2f} GiB)")
    sw = hdr["lm_head.weight_scale"]["data_offsets"][0]
    with open(src_st, "rb") as fin, open(dst_st, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for k in order:
            if k == WK:
                lo = hdr[k]["data_offsets"][0]
                for r0 in range(0, vocab, ROWS):
                    r1 = min(r0 + ROWS, vocab)
                    n = r1 - r0
                    fin.seek(data_start + lo + r0 * packed)
                    raw = np.frombuffer(fin.read(n * packed), dtype=np.uint8).reshape(n, packed)
                    fin.seek(data_start + sw + r0 * nblk)
                    sc = F8[np.frombuffer(fin.read(n * nblk), dtype=np.uint8)].reshape(n, nblk)
                    vals = np.stack([E2M1[raw & 0x0F], E2M1[raw >> 4]], axis=2).reshape(n, hidden)
                    vals *= np.repeat(sc, bs, axis=1) * s2
                    fout.write(f32_to_bf16_bits(vals).tobytes())
                    if r0 % (ROWS * 8) == 0:
                        print(f"  lm_head {100*r1/vocab:5.1f}%", flush=True)
            else:
                lo, hi = hdr[k]["data_offsets"]
                fin.seek(data_start + lo)
                left = hi - lo
                while left:
                    chunk = fin.read(min(left, 64 << 20))
                    fout.write(chunk)
                    left -= len(chunk)

    # Unchanged shards are hardlinked; JSON must be copied so we never touch SRC.
    for fn in os.listdir(SRC):
        if fn.startswith(".") or fn.endswith(".state") or fn == shard:
            continue
        s, d = os.path.join(SRC, fn), os.path.join(DST, fn)
        if not os.path.isfile(s) or os.path.exists(d):
            continue
        if fn.endswith(".safetensors"):
            os.link(s, d)
        else:
            shutil.copy2(s, d)

    idx_p = os.path.join(DST, "model.safetensors.index.json")
    idx = json.load(open(idx_p))
    for k in DROP:
        idx["weight_map"].pop(k, None)
    idx["metadata"]["total_size"] = sum(
        os.path.getsize(os.path.join(DST, f)) for f in set(idx["weight_map"].values()))
    json.dump(idx, open(idx_p, "w"), indent=2)

    cfg_p = os.path.join(DST, "config.json")
    cfg = json.load(open(cfg_p))
    q = cfg["quantization_config"]
    for g in q["config_groups"].values():
        g["targets"] = [t for t in g.get("targets", []) if t != "lm_head"]
    q.get("quantized_layers", {}).pop("lm_head", None)
    if "lm_head" not in q.get("ignore", []):
        q.setdefault("ignore", []).append("lm_head")
    json.dump(cfg, open(cfg_p, "w"), indent=2)

    print("done:", dst_st, f"{os.path.getsize(dst_st)/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
