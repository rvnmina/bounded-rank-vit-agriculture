"""
Bounded-Rank Attention — package init.
Convenience re-exports of the main model classes.
"""
from .core_model import (
    FourierFeatureEncoder,
    BoundedRankQuadraticAttention,
    TransformerBlock,
    BoundedRankVisionTransformer,
    StandardVisionTransformer,
)
from .fourier_domain import FourierBoundedRankAttention

__all__ = [
    "FourierFeatureEncoder",
    "BoundedRankQuadraticAttention",
    "TransformerBlock",
    "BoundedRankVisionTransformer",
    "StandardVisionTransformer",
    "FourierBoundedRankAttention",
]
