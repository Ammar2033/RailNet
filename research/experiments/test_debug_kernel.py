import importlib.util
import numpy as np
from pathlib import Path

HERE = Path("E:/Ammqr/Railnet")
def lm(p,n):
    spec=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
RN=lm(HERE/"04_bf16_learned_basis.py","rn")
R12=lm(HERE/"12_gemma_linear_runner.py","r12")

name="model.layers.0.self_attn.k_proj.weight"
RN.TARGET_TENSOR=name
raw,shape=RN.read_target_tensor()
out_f,in_f=int(shape[0]),int(shape[1])

comp=R12.CompiledTensor(HERE/"compiled/layer0/_GLOBAL_layer0.json", raw, shape)

rng=np.random.default_rng(7)
x=rng.normal(0,1,in_f).astype(np.float64)

W=(RN.bf16_array_to_float32(raw).astype(np.float64)).reshape(shape)
y_dense=x@W.T
y_rail=R12.rail_linear(x,comp)

diff=np.abs(y_rail-y_dense)
print("max|err|",diff.max(),"argmax",diff.argmax())
j=int(diff.argmax())

# manual grouped computation for this j
manual={}
Wj=W[j]
for i in range(in_f):
    g=int(raw[j*in_f+i])
    tot=0.0
    # find terms via tables
    for t in range(comp.max_terms):
        if comp.term_active[g,t]:
            r=comp.term_rail[g,t]; s=comp.term_sign[g,t]
            manual[r]=manual.get(r,0.0)+s*x[i]
    # sanity: reconstruct weight
    if abs(tot)>0: pass
# also verify per-element weight equality bit-level on row j
bad_w=0
for i in range(in_f):
    g=int(raw[j*in_f+i])
    tot=0.0
    for t in range(comp.max_terms):
        if comp.term_active[g,t]:
            tot+=comp.term_sign[g,t]*comp.rails_f64[comp.term_rail[g,t]]
    rec=RN.fp32_array_to_bf16_bits(np.array([tot]))[0]
    if rec!=np.uint16(g): bad_w+=1
print(f"row {j}: bad_weight_elems={bad_w}/{in_f}")
y_manual=np.zeros(out_f)
for i in range(in_f):
    pass
# manual y via per-element exact weight (should equal dense bitwise-ish)
y_exact=np.array([sum(float(W[j,i])*x[i] for i in range(in_f))])
print("y_dense[j]",y_dense[j],"y_rail[j]",y_rail[j])
print("grouped G for row j vs expected:")
g_row=raw[(j)*in_f:(j+1)*in_f].astype(np.int32)
Gm={}
for i in range(in_f):
    g=int(g_row[i])
    for t in range(comp.max_terms):
        if comp.term_active[g,t]:
            r=comp.term_rail[g,t]; s=comp.term_sign[g,t]
            Gm[r]=Gm.get(r,0.0)+s*x[i]
ym=sum(v*comp.rails_f64[r] for r,v in Gm.items())
print("ym_grouped_python",ym,"err_vs_dense",abs(ym-y_dense[j]))
