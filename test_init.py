import numpy as np, struct, json, mmap, math
from pathlib import Path

def bf16_bits_to_float32(bits): return np.float32(struct.unpack("<f", struct.pack("<I", int(bits)<<16))[0])
def bf16_array_to_float32(bits): return (bits.astype(np.uint32)<<16).view(np.float32)
def float32_to_bf16_bits(v): return struct.unpack("<I", struct.pack("<f", float(np.float32(v))))[0]>>16

MODEL_FILE=Path("E:/Ammqr/Railnet/model_data/model.safetensors")
TARGET_TENSOR="model.layers.0.mlp.up_proj.weight"

def read_safetensors_header():
    with open(MODEL_FILE,"rb") as f:
        hl=struct.unpack("<Q", f.read(8))[0]
        hb=f.read(hl)
    header=json.loads(hb.decode("utf-8"))
    return header, 8+hl

def read_target_tensor():
    header, base = read_safetensors_header()
    meta=header[TARGET_TENSOR]
    shape=tuple(int(x) for x in meta["shape"])
    off=meta["data_offsets"]
    start, end=int(off[0]), int(off[1])
    abs_start=base+start
    byte_count=end-start
    with open(MODEL_FILE,"rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            raw_bytes=mm[abs_start:abs_start+byte_count]
    raw=np.frombuffer(raw_bytes, dtype=np.uint16).copy()
    return raw, shape

def analyze_unique(raw):
    uniq, counts=np.unique(raw, return_counts=True)
    vals=bf16_array_to_float32(uniq).astype(np.float64)
    return uniq, counts.astype(np.float64), vals

# Load
raw, shape = read_target_tensor()
uniq_bits, counts, vals = analyze_unique(raw)
print(f"unique {len(uniq_bits)} vals range {vals.min():.6f} {vals.max():.6f}")

# Current initialize logic using bits as values
def initialize_rails_current(values, counts, rail_count):
    if rail_count >= len(values):
        selected=values.copy()
        if len(selected)<rail_count:
            padding=np.zeros(rail_count-len(selected), dtype=np.uint16)
            selected=np.concatenate([selected, padding])
        return selected[:rail_count]
    order=np.argsort(values)
    sorted_values=values[order]
    sorted_counts=counts[order]
    cumulative=np.cumsum(sorted_counts)
    total=cumulative[-1]
    centers=[]
    for i in range(rail_count):
        q=(i+0.5)/rail_count
        target=q*total
        idx=int(np.searchsorted(cumulative, target, side="left"))
        idx=min(idx, len(sorted_values)-1)
        centers.append(float(sorted_values[idx]))
    centers=np.asarray(centers,dtype=np.float64)
    for _ in range(8):
        distance=np.abs(values[:,None]-centers[None,:])
        assign=np.argmin(distance,axis=1)
        new=centers.copy()
        for k in range(rail_count):
            mask=assign==k
            if not np.any(mask): continue
            new[k]=np.sum(values[mask]*counts[mask])/max(np.sum(counts[mask]),1.0)
        centers=new
    rails=np.array([float32_to_bf16_bits(c) for c in centers], dtype=np.uint16)
    rails=np.unique(rails)
    if len(rails)<rail_count:
        freq_order=np.argsort(counts)[::-1]
        used=set(int(x) for x in rails)
        adds=[]
        for idx in freq_order:
            cand=int(values[idx])
            if cand in used: continue
            used.add(cand)
            adds.append(cand)
            if len(rails)+len(adds)>=rail_count: break
        if adds:
            rails=np.concatenate([rails, np.array(adds,dtype=np.uint16)])
    return rails[:rail_count]

# Fixed version
def initialize_rails_fixed(values_float, bits, counts, rail_count):
    if rail_count >= len(values_float):
        selected=bits.copy()
        if len(selected)<rail_count:
            padding=np.zeros(rail_count-len(selected), dtype=np.uint16)
            selected=np.concatenate([selected, padding])
        return selected[:rail_count]
    order=np.argsort(values_float)
    sorted_values=values_float[order]
    sorted_counts=counts[order]
    cumulative=np.cumsum(sorted_counts)
    total=cumulative[-1]
    centers=[]
    for i in range(rail_count):
        q=(i+0.5)/rail_count
        target=q*total
        idx=int(np.searchsorted(cumulative, target, side="left"))
        idx=min(idx, len(sorted_values)-1)
        centers.append(float(sorted_values[idx]))
    centers=np.asarray(centers,dtype=np.float64)
    for _ in range(8):
        distance=np.abs(values_float[:,None]-centers[None,:])
        assign=np.argmin(distance,axis=1)
        new=centers.copy()
        for k in range(rail_count):
            mask=assign==k
            if not np.any(mask): continue
            new[k]=np.sum(values_float[mask]*counts[mask])/max(np.sum(counts[mask]),1.0)
        centers=new
    rails=np.array([float32_to_bf16_bits(c) for c in centers], dtype=np.uint16)
    rails=np.unique(rails)
    if len(rails)<rail_count:
        freq_order=np.argsort(counts)[::-1]
        used=set(int(x) for x in rails)
        adds=[]
        for idx in freq_order:
            cand=int(bits[idx])
            if cand in used: continue
            used.add(cand)
            adds.append(cand)
            if len(rails)+len(adds)>=rail_count: break
        if adds:
            rails=np.concatenate([rails, np.array(adds,dtype=np.uint16)])
    return rails[:rail_count]

curr = initialize_rails_current(uniq_bits, counts, 32)
print("current rails bits:", curr[:10])
print("current decoded:", [float(bf16_bits_to_float32(b)) for b in curr[:5]])
# check values magnitude: should be around -0.5 to 0.6, but current will be large if bug
fixed = initialize_rails_fixed(vals, uniq_bits, counts, 32)
print("fixed rails bits:", fixed[:10])
print("fixed decoded:", [float(bf16_bits_to_float32(b)) for b in fixed[:10]])
# Compare distances: compute weighted MSE for dummy?
print("curr unique?", len(np.unique(curr)), "fixed unique?", len(np.unique(fixed)))
# Show sorted vals vs curr centers conceptual
# Also test calling with values=bits vs values_float
# Let's also look at what center values would be if interpreting bits as float numbers
# bits max ~ 0xBF00? That's 48960 decimal => float 48960 => bf16 of that is huge
