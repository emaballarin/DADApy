import sys,os
os.environ["JAX_ENABLE_X64"] = "True" # Allowing double precision on JAX before any JAX import

from ._utils.utils import *
from .base import Base
from .clustering import Clustering
from .data import Data
from .data_sets import DataSets
from .density_advanced import DensityAdvanced
from .density_estimation import DensityEstimation
from .feature_weighting import FeatureWeighting
from .id_discrete import *
from .id_estimation import IdEstimation
from .kstar import KStar
from .metric_comparisons import MetricComparisons
from .neigh_graph import NeighGraph
from .hamming import Hamming, BID

if sys.version_info >= (3, 9):
    from .diff_imbalance import DiffImbalance
if sys.version_info >= (3, 9):
    from .causal_graph import CausalGraph
