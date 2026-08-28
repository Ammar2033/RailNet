import numpy as np, struct

def bf16_bits_to_float32(bits):
    return np.float32(struct.unpack("<f", struct.pack("<I", int(bits)<<16))[0])
def float32_to_bf16_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(np.float32(v))))[0] >>16
def bf16_array_to_float32(bits):
    return (bits.astype(np.uint32)<<16).view(np.float32)

rails = np.array([0x3F80, 0x3F00, 0x3E80, 0x3D80], dtype=np.uint16)
print([float(bf16_bits_to_float32(b)) for b in rails])

target_values = np.array([1.0, 0.75, 0.5, 0.25], dtype=np.float64)

def greedy_current(target_values, rails, max_terms):
    n=len(target_values); r=len(rails)
    rail_values=bf16_array_to_float32(rails).astype(np.float64)
    residual=target_values.copy()
    routes=np.zeros((n,max_terms),dtype=np.int16)
    signs=np.zeros((n,max_terms),dtype=np.int8)
    used=np.zeros((n,r),dtype=bool)
    active=np.zeros(n,dtype=np.int8)
    for term in range(max_terms):
        pos=np.abs(residual[:,None]-rail_values[None,:])
        neg=np.abs(residual[:,None]+rail_values[None,:])
        choose=pos<=neg
        err=np.minimum(pos,neg)
        err[used]=np.inf
        best=np.argmin(err,axis=1)
        rows=np.arange(n)
        best_err=err[rows,best]
        valid=np.isfinite(best_err)
        s=np.where(choose[rows,best],1,-1).astype(np.int8)
        routes[rows[valid],term]=best[valid]+1
        signs[rows[valid],term]=s[valid]
        sel=rail_values[best]
        residual[valid]-=s[valid]*sel[valid]
        used[rows[valid],best[valid]]=True
        active[valid]+=1
        print(f"term {term}: routes {routes[:,term]} signs {signs[:,term]} residual {residual}")
    return routes,signs,residual,active

r,s,res,act=greedy_current(target_values, rails, 4)
print("final residual",res)

def reconstruct(routes, signs, rails):
    rv=bf16_array_to_float32(rails).astype(np.float64)
    rec=np.zeros(routes.shape[0],dtype=np.float64)
    for t in range(routes.shape[1]):
        ids=routes[:,t]
        act2=ids>0
        if not np.any(act2): continue
        zb=ids[act2]-1
        rec[act2]+=signs[act2,t].astype(np.float64)*rv[zb]
    return rec

rec=reconstruct(r,s,rails)
print("targets",target_values)
print("rec",rec)
print("diff",target_values-rec)

print("\n--- test exact single rail ---")
target2=np.array([float(bf16_bits_to_float32(0x3F80))],dtype=np.float64)
print("target2",target2)
r2,s2,res2,act2=greedy_current(target2, rails, 4)
print(r2,s2,res2,act2)
rec2=reconstruct(r2,s2,rails)
print("rec2",rec2, "diff",target2-rec2)
print("target bits",hex(float32_to_bf16_bits(target2[0])), "rec bits",hex(float32_to_bf16_bits(rec2[0])))
