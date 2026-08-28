import sys
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from railnet.compiler import RailNetCompiler
from railnet.artifacts import write_rnmodel
from railnet.safetensors_reader import list_tensors, read_tensor_raw, tensor_metadata

def main():
    print("="*60)
    print("RAILNET MODERN COMPILER PIPELINE (STAGE 15A)")
    print("="*60)
    
    compiler = RailNetCompiler(model="gemma3", default_dtype="bf16")
    
    tensors = []
    route_maps = {}
    
    # We will just compile a tiny subset for the test
    target_tensors = [
        "model.layers.0.self_attn.o_proj.weight"
    ]
    
    for name in target_tensors:
        print(f"Compiling {name}...")
        raw, shape = read_tensor_raw(name)
        
        # We limit the shape to just 128x128 to make the test fast
        raw = raw[:128*128]
        shape = (128, 128)
        
        compiled_tensor = compiler.compile_tensor(
            raw, 
            dtype="bf16", 
            rails=96, 
            max_terms=4, 
            name=name,
            shape=shape
        )
        
        tensors.append(compiled_tensor.to_dict())
        route_maps[name] = compiled_tensor.route_ids
        
    out_path = HERE.parent.parent / "compiled" / "gemma3_compressed.rnmodel"
    write_rnmodel(str(out_path), "gemma3", "bf16", tensors, route_maps)
    
    print(f"Saved compressed artifact to {out_path}")

if __name__ == "__main__":
    main()
