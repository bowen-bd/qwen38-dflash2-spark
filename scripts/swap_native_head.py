#!/usr/bin/env python3
"""Swap in the *native* BF16 lm_head instead of the NVFP4-dequantized one.

patch_lmhead_nvfp4.py makes the head dense but not accurate: it reproduces the
NVFP4 values exactly, so it carries NVFP4's quantization error (cosine 0.9955
against the true weights). The official FP8 checkpoint ships lm_head as genuinely
unquantized BF16 -- quant_method fp8, lm_head absent from quantized_layers -- i.e.
the original tensor. Same shape [248320, 5120] and same dtype, so it overwrites in
place at the same offset and the safetensors header needs no change.

Strictly better for DFlash2: its selector reads the head to pick top-K candidates,
so a faithful head should also improve acceptance, not just logits.
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

HOME = os.environ.get("QWEN_HOME", os.path.expanduser("~/llm"))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "radixark-nvfp4-dflash2")
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "qwen38-27b-fp8")
DST = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HOME, "radixark-nvfp4-bf16head")
WK = "lm_head.weight"


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def find(root, key):
    """Locate (file, byte offset, nbytes, dtype, shape) for one tensor."""
    idx = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(idx):
        shard = json.load(open(idx))["weight_map"][key]
    else:
        shard = next(f for f in os.listdir(root) if f.endswith(".safetensors"))
    p = os.path.join(root, shard)
    hdr, ds = read_header(p)
    e = hdr[key]
    lo, hi = e["data_offsets"]
    return p, ds + lo, hi - lo, e["dtype"], e["shape"]


def main():
    os.makedirs(DST, exist_ok=True)
    sp, so, sn, sd, ss = find(SRC, WK)
    rp, ro, rn, rd, rs = find(REF, WK)
    print(f"target head: {os.path.basename(sp)} dtype={sd} shape={ss} bytes={sn}")
    print(f"native head: {os.path.basename(rp)} dtype={rd} shape={rs} bytes={rn}")
    assert (sd, ss, sn) == (rd, rs, rn), "shape/dtype/size mismatch -- cannot swap in place"

    shard = os.path.basename(sp)
    for fn in os.listdir(SRC):
        if fn.startswith(".") or fn.endswith(".state"):
            continue
        s, d = os.path.join(SRC, fn), os.path.join(DST, fn)
        if not os.path.isfile(s) or os.path.exists(d):
            continue
        if fn == shard:
            print(f"copying {fn} ({os.path.getsize(s)/2**30:.2f} GiB)")
            shutil.copy2(s, d)
        elif fn.endswith(".safetensors"):
            os.link(s, d)
        else:
            shutil.copy2(s, d)

    dp = os.path.join(DST, shard)
    print("overwriting lm_head in place")
    with open(rp, "rb") as fin, open(dp, "r+b") as fout:
        fin.seek(ro)
        fout.seek(so)
        left = sn
        while left:
            chunk = fin.read(min(left, 64 << 20))
            fout.write(chunk)
            left -= len(chunk)

    # Verify against the reference and against the dequantized version.
    R0, R1 = 1000, 1512
    hidden = ss[1]

    def rows(path, off):
        with open(path, "rb") as f:
            f.seek(off + R0 * hidden * 2)
            raw = np.frombuffer(f.read((R1 - R0) * hidden * 2), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)

    def cos(a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    new, ref, old = rows(dp, so), rows(rp, ro), rows(sp, so)
    print(f"  new vs native reference : {cos(new, ref):.6f}   (want 1.000000)")
    print(f"  new vs NVFP4-dequantized: {cos(new, old):.6f}   (want ~0.9955)")
    print("done:", dp)


if __name__ == "__main__":
    main()
