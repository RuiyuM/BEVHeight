from .uncertainty import UncertaintySelector
from .random import RandomSelector
from .entropy import EntropySelector
from .label_diversity import LabelDiversityHighSelector, LabelDiversityLowSelector
from .geo_bias import FarBiasSelector, NearOccupancySelector
from .geo_gauss import GaussSimSelector, GaussAntiSimSelector

_METHOD_REGISTRY = {
    "uncertainty": UncertaintySelector,
    "random": RandomSelector,
    "entropy": EntropySelector,
    "label_diversity_high": LabelDiversityHighSelector,
    "label_diversity_low": LabelDiversityLowSelector,
    "far_bias": FarBiasSelector,
    "near_occupancy": NearOccupancySelector,
    "gauss_sim": GaussSimSelector,   # 选最接近当前标注分布的
    "gauss_anti": GaussAntiSimSelector, # 选最不接近的
}

def get_method(name: str):
    name = (name or "uncertainty").lower()
    if name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown active method '{name}'. Options: {list(_METHOD_REGISTRY.keys())}")
    return _METHOD_REGISTRY[name]