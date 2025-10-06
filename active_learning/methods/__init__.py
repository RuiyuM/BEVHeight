from .uncertainty import UncertaintySelector

_METHOD_REGISTRY = {
    "uncertainty": UncertaintySelector,
}

def get_method(name: str):
    name = (name or "uncertainty").lower()
    if name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown active method '{name}'. Options: {list(_METHOD_REGISTRY.keys())}")
    return _METHOD_REGISTRY[name]

