import importlib.util, json, numpy as np, struct
spec = importlib.util.spec_from_file_location("mod", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Reader A: 04's mmap reader
rawA, shapeA = mod.read_target_tensor()

# Reader B: replicate 05's full-read
MODEL_FILE = "E:/Ammqr/Railnet/model_data/model.safetensors"
TARGET="model.layers.0.mlp.up_proj.weight"
with open(MODEL_FILE,"rb") as f:
    hl=struct.unpack("<Q", f.read(8))[0]
    header=json.loads(f.read(hl).decode("utf-8"))
    base=8+hl
    meta=header[TARGET]
    s,e=meta["data_offsets"]
    abs_=base+s
    bc=e-s
    rest=f.read()
rawB=np.frombuffer(rest[abs_:abs_+bc], dtype=np.uint16).copy()
shapeB=tuple(meta["shape"])

print("A:", len(rawA), shapeA, "uniques", len(np.unique(rawA)))
print("B:", len(rawB), shapeB, "uniques", len(np.unique(rawB)))
same = np.array_equal(rawA, rawB)
print("identical?", same)
if not same:
    diff=np.flatnonzero(rawA!=rawB)
    print("first diffs at", diff[:5], "count", len(diff))
