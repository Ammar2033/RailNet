import importlib.util, json
spec = importlib.util.spec_from_file_location("rn", "E:/Ammqr/Railnet/04_bf16_learned_basis.py")
RN = importlib.util.module_from_spec(spec); spec.loader.exec_module(RN)

header, base = RN.read_safetensors_header()
print("total tensors:", len([k for k in header if k != "__metadata__"]))

cats = {}
lm_head_present = False
for k in header:
    if k == "__metadata__": continue
    if k.startswith("model.layers."):
        parts = k.split(".")
        li = int(parts[2])
        sub = ".".join(parts[3:])
        key = (sub.split(".")[0] if len(parts)>4 else sub)
    elif k.startswith("model.embed_tokens"): key = "EMBED"
    elif k.startswith("lm_head"): key = "LM_HEAD"; lm_head_present=True
    elif k.startswith("model.norm"): key = "FINAL_NORM"
    else: key = "OTHER:"+k
    cats.setdefault(key, 0)
    cats[key]+=1

for k in sorted(cats): print(f"  {k:12s} x{cats[k]}")
print("lm_head present:", lm_head_present)

# sample names per group
seen=set()
for k in sorted(header):
    if k=="__metadata__": continue
    g=None
    if ".self_attn." in k: g="attn"
    elif ".mlp." in k: g="mlp"
    elif "layernorm" in k or k.startswith("model.norm") or "_norm." in k: g="norm"
    elif k.startswith("model.embed_tokens"): g="embed"
    elif k.startswith("lm_head"): g="lm_head"
    if g and g not in seen and ("layers.1." in k or g in("embed","lm_head")):
        print(f"sample[{g}]:", k, header[k]["shape"], header[k]["dtype"])
        seen.add(g)
