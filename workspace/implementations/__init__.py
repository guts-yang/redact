"""Decoupled reference implementations for IDEA_01..IDEA_10."""
from .bound import BOUND, BOUNDConfig
from .cfa import CFA, CFAConfig
from .csda import CSDA, CSDAConfig
from .disco import DISCO, DISCOConfig
from .eas import EAS, EASConfig
from .ortho import ORTHO, ORTHOConfig
from .ripple import RIPPLE, RIPPLEConfig
from .rrbu import RRBU, RRBUConfig
from .scope import SCOPE, SCOPEConfig
from .sequ import SEQU, SEQUConfig

__all__ = [
    "BOUND", "BOUNDConfig",
    "CFA", "CFAConfig",
    "CSDA", "CSDAConfig",
    "DISCO", "DISCOConfig",
    "EAS", "EASConfig",
    "ORTHO", "ORTHOConfig",
    "RIPPLE", "RIPPLEConfig",
    "RRBU", "RRBUConfig",
    "SCOPE", "SCOPEConfig",
    "SEQU", "SEQUConfig",
]
