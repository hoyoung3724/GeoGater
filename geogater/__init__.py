"""GeoGater: meta-feature guided adaptive routing for worldwide image geolocalization."""

from .meta_features import FEATURE_NAMES, G3_MASK, build_meta_feature_vector
from .router import ROUTE_RET, ROUTE_ZS, GeoGater, GeoLocLoss

__all__ = [
    "GeoGater",
    "GeoLocLoss",
    "ROUTE_ZS",
    "ROUTE_RET",
    "FEATURE_NAMES",
    "G3_MASK",
    "build_meta_feature_vector",
]
__version__ = "1.0.0"
