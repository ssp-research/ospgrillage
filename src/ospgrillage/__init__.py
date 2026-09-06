import warnings as _warnings

import numpy as np  # re-exported: tests and users access ospgrillage.np
import openseespy.opensees as ops  # re-exported: tests and users access ospgrillage.ops
import opsvis as opsv  # used internally by postprocessing (section_force_distribution_3d)
import matplotlib.pyplot as plt  # re-exported: users access ospgrillage.plt
from ospgrillage.utils import *
from ospgrillage.mesh import *
from ospgrillage.load import *
from ospgrillage.material import *
from ospgrillage.members import *
from ospgrillage.osp_grillage import *
from ospgrillage.postprocessing import *

__version__ = "0.6.0"

# Explicit public API — everything a user should access from `import ospgrillage`
__all__ = [
    "__version__",
    # Grillage model
    "OspGrillage",
    "OspGrillageBeam",
    "OspGrillageShell",
    "create_grillage",
    "InfluenceResultSet",
    "InfluenceLineResults",
    "InfluenceSurfaceResults",
    # Members & sections
    "GrillageMember",
    "Section",
    "create_member",
    "create_section",
    # Materials
    "Material",
    "create_material",
    # Loads
    "LoadCase",
    "LoadModel",
    "Loads",
    "MovingLoad",
    "NodalLoad",
    "NodeForces",
    "PatchLoading",
    "Path",
    "PointLoad",
    "LineLoading",
    "LoadVertex",
    "LoadPoint",  # deprecated alias for LoadVertex
    "CompoundLoad",
    "Line",
    "ShapeFunction",
    "create_load_vertex",
    "create_load",
    "create_load_case",
    "create_load_model",
    "create_moving_load",
    "create_moving_path",
    "create_compound_load",
    # Mesh / geometry
    "Point",
    "Mesh",
    "create_point",
    # Post-processing & plotting
    "Envelope",
    "InfluenceLine",
    "InfluenceSurface",
    "Members",
    "PostProcessor",
    "create_envelope",
    "model_proxy_from_results",
    "create_influence_line",
    "create_influence_surface",
    "plot_il",
    "plot_is",
    "plot_force",
    "plot_bmd",
    "plot_sfd",
    "plot_tmd",
    "plot_def",
    "plot_model",
    "plot_srf",
    "plot_load_plan",
]


# ---------------------------------------------------------------------------
# Backwards-compatible lazy access for deprecated re-exports
# ---------------------------------------------------------------------------
# Hide opsv from the public namespace — it is imported above for internal use
# by postprocessing.py but should not be accessed as og.opsv by users.
def __getattr__(name):
    if name == "opsv":
        _warnings.warn(
            "og.opsv is deprecated — use og.plot_model() for mesh visualisation. "
            "Direct opsv access will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return opsv
    if name == "opsplt":
        _warnings.warn(
            "og.opsplt is deprecated — use og.plot_model() for mesh visualisation. "
            "vfo/opsplt will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import vfo.vfo as _opsplt

            return _opsplt
        except ImportError:
            raise ImportError(
                "vfo is no longer a required dependency. "
                "Install it with: pip install vfo"
            ) from None
    raise AttributeError(f"module 'ospgrillage' has no attribute {name!r}")
