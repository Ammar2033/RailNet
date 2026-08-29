"""Norm — passthrough (RMSNorm weights are small vectors, kept dense)."""


def compile_norm(weight, name=""):
    return {
        "status": "PASSTHROUGH",
        "name": name,
        "shape": list(weight.shape) if hasattr(weight, "shape") else None,
    }
