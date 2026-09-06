# -*- coding: utf-8 -*-
"""
This module contains functions and classes related to post processing processes.
The post processing module is an addition to the currently available post processing
module of OpenSeesPy - this module fills in gaps to
* create envelope from xarray DataSet
* plot force and deflection diagrams from xarray DataSets
"""

import enum
import json
import matplotlib.pyplot as plt
import opsvis as opsv
import numpy as np
import re
from typing import TYPE_CHECKING, Union
import xarray as xr

# if TYPE_CHECKING:
from ospgrillage.load import ShapeFunction
from ospgrillage.utils import solve_zeta_eta
from ospgrillage.load_plot import plot_load_plan

__all__ = [
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
# Lightweight model proxy for standalone results files
# ---------------------------------------------------------------------------
class _ModelProxy:
    """Reconstruct the model interface needed by plot functions from a Dataset.

    When results are saved to NetCDF with
    :meth:`~ospgrillage.osp_grillage.OspGrillage.get_results` the file
    contains ``node_coordinates``, ``member_elements`` (JSON attr), and
    ``model_type`` (attr).  This class wraps a loaded Dataset and exposes
    the same ``get_nodes`` / ``get_element`` /
    ``common_grillage_element_z_group`` interface that the plotting helpers
    expect.
    """

    def __init__(self, ds):
        self.model_type = ds.attrs.get("model_type", "beam_link")

        # Build node_spec dict: {tag: {"coordinate": [x, y, z]}}
        coords_da = ds["node_coordinates"]
        # Combined influence datasets may carry an extra leading study dimension.
        # Geometry is identical across studies, so take the first slice.
        for dim in list(coords_da.dims):
            if dim not in {"Node", "Axis"}:
                coords_da = coords_da.isel({dim: 0})
        self._node_spec = {}
        for tag in coords_da.coords["Node"].values:
            self._node_spec[int(tag)] = {
                "coordinate": coords_da.sel(Node=tag).values.tolist()
            }

        # Build member-element mapping from JSON attr
        self._members = json.loads(ds.attrs.get("member_elements", "{}"))

        # Optional beam element connectivity (element -> [node_i, node_j]) for
        # robust model rendering from saved files even when member node lists are
        # not ordered along the geometric line.
        self._element_nodes = {}
        ele_nodes = ds.get("ele_nodes", None)
        if ele_nodes is not None:
            ele_da = ele_nodes
            for dim in list(ele_da.dims):
                if dim not in {"Element", "Nodes"}:
                    ele_da = ele_da.isel({dim: 0})
            if {"Element", "Nodes"}.issubset(set(ele_da.dims)):
                for ele_tag in ele_da.coords["Element"].values:
                    node_vals = np.asarray(ele_da.sel(Element=ele_tag).values).tolist()
                    node_tags = []
                    for node_val in node_vals:
                        try:
                            node_f = float(node_val)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(node_f):
                            continue
                        node_tags.append(int(node_f))
                    if len(node_tags) >= 2:
                        self._element_nodes[int(ele_tag)] = tuple(node_tags[:2])

        # Optional shell connectivity (for shell-beam model visualisation when
        # plotting from saved results files without a live OspGrillage object).
        self._shell_element_nodes = []
        ele_nodes_shell = ds.get("ele_nodes_shell", None)
        if ele_nodes_shell is not None:
            shell_da = ele_nodes_shell
            for dim in list(shell_da.dims):
                if dim not in {"Element", "Nodes"}:
                    shell_da = shell_da.isel({dim: 0})
            if {"Element", "Nodes"}.issubset(set(shell_da.dims)):
                for node_row in np.asarray(shell_da.values):
                    node_tags = []
                    for node_tag in np.asarray(node_row).tolist():
                        if node_tag is None:
                            continue
                        try:
                            node_val = float(node_tag)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(node_val):
                            continue
                        node_tags.append(int(node_val))
                    if len(node_tags) >= 3:
                        self._shell_element_nodes.append(tuple(node_tags))

        # Reconstruct common_grillage_element_z_group (member → list of
        # z-group indices, e.g. {"interior_main_beam": [0, 1, 2]}).
        self.common_grillage_element_z_group = {}
        for member, info in self._members.items():
            self.common_grillage_element_z_group[member] = list(
                range(len(info["elements"]))
            )

    def get_nodes(self, number=None):
        """Return node specification dict, or coordinates of a single node."""
        if number:
            return self._node_spec[number]["coordinate"]
        return self._node_spec

    def get_element(self, **kwargs):
        """Return element tags or node tags for a member group."""
        member = kwargs.get("member", None)
        options = kwargs.get("options", "nodes")
        z_group_num = kwargs.get("z_group_num", 0)

        info = self._members.get(member, {"elements": [[]], "nodes": [[]]})
        key = "elements" if options == "elements" else "nodes"
        groups = info.get(key, [[]])
        if z_group_num < len(groups):
            return groups[z_group_num]
        return []


def model_proxy_from_results(ds):
    """Create a lightweight model proxy from a self-contained results Dataset.

    The proxy satisfies the interface required by plotting functions
    (:func:`plot_bmd`, :func:`plot_sfd`, etc.) so that results saved to
    NetCDF can be visualised without the original
    :class:`~ospgrillage.osp_grillage.OspGrillage` object.

    :param ds: Dataset loaded via ``xarray.open_dataset()``.  Must contain
        a ``node_coordinates`` variable and ``member_elements`` /
        ``model_type`` attributes (added automatically by
        :meth:`~ospgrillage.osp_grillage.OspGrillage.get_results`).
    :type ds: :class:`xarray.Dataset`
    :returns: A proxy object with ``get_nodes()``, ``get_element()``, and
        ``common_grillage_element_z_group`` matching the OspGrillage
        interface.
    :raises KeyError: If the Dataset is missing the required geometry data.

    Example::

        import xarray as xr
        import ospgrillage as og

        ds = xr.open_dataset("results.nc")
        proxy = og.model_proxy_from_results(ds)
        og.plot_bmd(proxy, ds, backend="plotly")
    """
    if "node_coordinates" not in ds:
        raise KeyError(
            "Dataset does not contain 'node_coordinates'. "
            "Re-save results with ospgrillage >= 0.5.4 to include geometry."
        )
    return _ModelProxy(ds)


# Sentinel for auto-generated titles.  ``title=_AUTO`` means "use the
# default"; ``title=None`` means "no title"; ``title="..."`` is a custom
# override.
_AUTO = object()
_LOADCASE_POSITION_RE = re.compile(
    r"^(?P<name>.+) at global position \[(?P<x>[-+\d.eE]+),(?P<y>[-+\d.eE]+),(?P<z>[-+\d.eE]+)\]$"
)


# ---------------------------------------------------------------------------
# Lazy import for optional plotly dependency
# ---------------------------------------------------------------------------
def _import_plotly():
    """Lazy-import plotly; raise a clear error if missing."""
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install it with: pip install ospgrillage[gui]"
        )


def _show_plotly_fig(fig):
    """Display a Plotly figure.

    In a Jupyter *notebook* session this emits ``text/html`` using
    ``fig.to_html(include_plotlyjs="cdn")``.  The ``text/html`` MIME type
    is understood by *nbsphinx* so the interactive 3-D plot survives the
    Sphinx build and renders with full rotate/zoom/hover on the
    documentation pages (e.g. GitHub Pages).

    In an IPython terminal (no notebook) or outside IPython the function
    falls back to ``fig.show()`` which opens the system browser.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            # Jupyter notebook — use HTML display for nbsphinx compat
            from IPython.display import display, HTML

            display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))
            return
    except ImportError:
        pass
    # IPython terminal or plain Python — open in browser
    fig.show()


def create_envelope(**kwargs):
    """
    Create an envelope object for post-processing result envelopes.

    The constructor takes an `xarray` DataSet and kwargs for enveloping options.

    :param ds: Result DataSet from :func:`~ospgrillage.osp_grillage.OspGrillage.get_results`.
    :type ds: xarray.Dataset
    :param load_effect: Specific load effect component to envelope.
    :type load_effect: str, optional
    :param array: Data array to envelope — either ``"displacements"`` or ``"forces"``.
    :type array: str, optional
    :param value_mode: If ``True``, return raw envelope values. Defaults to ``True``.
    :type value_mode: bool, optional
    :param query_mode: If ``True``, return the load case coordinates at the envelope extrema. Defaults to ``False``.
    :type query_mode: bool, optional
    :param extrema: Envelope direction — either ``"min"`` or ``"max"``.
    :type extrema: str, optional
    :param elements: Specific element tags to include in the envelope.
    :type elements: list, optional
    :param nodes: Specific node tags to include in the envelope.
    :type nodes: list, optional
    :returns: :class:`Envelope` object.
    """
    return Envelope(**kwargs)


def create_influence_line(**kwargs):
    """
    Create an influence line object from stored moving-load or grid-load results.

    The helper extracts one response quantity from an ``xarray.Dataset`` and
    reindexes it against one load-position coordinate, typically ``x`` for a
    driving-lane influence line.

    :param ds: Result DataSet from :func:`~ospgrillage.osp_grillage.OspGrillage.get_results`.
    :type ds: xarray.Dataset
    :param component: Specific response component to extract.
    :type component: str
    :param array: Data array to query, e.g. ``"displacements"``, ``"forces"``,
        ``"forces_beam"``, or ``"forces_shell"``.
    :type array: str, optional
    :param load_coord: Load-position coordinate to use as the influence-line axis.
        One of ``"x"``, ``"y"``, ``"z"``, or cumulative path ``"station"``.
        Defaults to ``"x"``.
    :type load_coord: str, optional
    :param loadcase: Optional load case name or list of names to include.
    :type loadcase: str or list[str], optional
    :param values: Optional explicit ``(x, y, z)`` load positions corresponding to
        the selected load cases. If omitted, positions are parsed from the
        ``Loadcase`` coordinate strings.
    :type values: list[tuple[float, float, float]], optional
    :param node: Optional node tag or list of node tags to select.
    :type node: int or list[int], optional
    :param element: Optional element tag or list of element tags to select.
    :type element: int or list[int], optional
    :returns: :class:`InfluenceLine` object.
    """
    return InfluenceLine(**kwargs)


def create_influence_surface(**kwargs):
    """
    Create an influence surface object from stored results.

    The helper extracts one response quantity from an ``xarray.Dataset`` and
    reshapes it onto a 2D load-position grid, typically ``x`` by ``z`` for
    bridge deck influence surfaces.

    :param ds: Result DataSet from :func:`~ospgrillage.osp_grillage.OspGrillage.get_results`.
    :type ds: xarray.Dataset
    :param component: Specific response component to extract.
    :type component: str
    :param array: Data array to query, e.g. ``"displacements"``, ``"forces"``,
        ``"forces_beam"``, or ``"forces_shell"``.
    :type array: str, optional
    :param x_coord: Load-position coordinate to use on the first surface axis.
        Defaults to ``"x"``. Station-based datasets may also use
        ``"longitudinal_station"`` or ``"transverse_station"``.
    :type x_coord: str, optional
    :param y_coord: Load-position coordinate to use on the second surface axis.
        Defaults to ``"z"``. Station-based datasets may also use
        ``"longitudinal_station"`` or ``"transverse_station"``.
    :type y_coord: str, optional
    :param loadcase: Optional load case name or list of names to include.
    :type loadcase: str or list[str], optional
    :param values: Optional explicit ``(x, y, z)`` load positions corresponding to
        the selected load cases. If omitted, positions are parsed from the
        ``Loadcase`` coordinate strings.
    :type values: list[tuple[float, float, float]], optional
    :param node: Optional node tag or list of node tags to select.
    :type node: int or list[int], optional
    :param element: Optional element tag or list of element tags to select.
    :type element: int or list[int], optional
    :returns: :class:`InfluenceSurface` object.
    """
    return InfluenceSurface(**kwargs)


def _parse_loadcase_position(loadcase_name):
    match = _LOADCASE_POSITION_RE.match(str(loadcase_name))
    if not match:
        raise ValueError(
            "Unable to determine load position from Loadcase={!r}. "
            "Expected names like '<load name> at global position [x,y,z]' "
            "or pass values=[(x, y, z), ...].".format(loadcase_name)
        )
    return (
        float(match.group("x")),
        float(match.group("y")),
        float(match.group("z")),
    )


def _normalise_loadcase_selection(loadcase):
    if loadcase is None:
        return None
    if isinstance(loadcase, str):
        return [loadcase]
    return list(loadcase)


def _select_response_data(ds, array, component, node=None, element=None, loadcase=None):
    da = getattr(ds, array)
    sel_kwargs = {"Component": component}
    if "Node" in da.dims and node is not None:
        sel_kwargs["Node"] = node
    if "Element" in da.dims and element is not None:
        sel_kwargs["Element"] = element
    if loadcase is not None:
        sel_kwargs["Loadcase"] = _normalise_loadcase_selection(loadcase)
    return da.sel(**sel_kwargs)


def _get_position_index(da, values=None):
    loadcases = da.coords["Loadcase"].values.tolist()
    if values is None:
        if all(
            coord in da.coords
            for coord in ("load_position_x", "load_position_y", "load_position_z")
        ):
            positions = list(
                zip(
                    da.coords["load_position_x"].values.tolist(),
                    da.coords["load_position_y"].values.tolist(),
                    da.coords["load_position_z"].values.tolist(),
                )
            )
        else:
            positions = [_parse_loadcase_position(name) for name in loadcases]
    else:
        positions = [tuple(map(float, position)) for position in values]
        if len(positions) != len(loadcases):
            raise ValueError(
                "values= must have the same length as the selected Loadcase coordinate"
            )
    return xr.DataArray(
        np.asarray(positions, dtype=float),
        dims=("Loadcase", "position_component"),
        coords={
            "Loadcase": da.coords["Loadcase"].values,
            "position_component": ["x", "y", "z"],
        },
        name="load_position",
    )


def _cumulative_path_station(position_index):
    """Return cumulative path distance for each load position."""
    coords = np.asarray(position_index.values, dtype=float)
    if len(coords) == 0:
        return np.asarray([], dtype=float)
    if len(coords) == 1:
        return np.asarray([0.0], dtype=float)
    deltas = np.diff(coords, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(deltas, axis=1))])


class Envelope:
    """
    Class for defining envelope of :class:`~ospgrillage.osp_grillage.OspGrillage`'s
    `xarray` of result.

    A :func:`Envelope.get` method is provided that returns an enveloped `xarray` based
    on the specified input parameters of this class.


    """

    def __init__(self, ds, load_effect: str = None, **kwargs):
        """

        :param ds: Data set from
                   :func:`~ospgrillage.osp_grillage.OspGrillage.get_results` . note Combination
        :type ds: Xarray
        :param load_effect: Specific load effect to envelope.
        :type load_effect: str
        :param array: Data array to envelope — either ``"displacements"`` or ``"forces"``.
        :type array: str, optional
        :param value_mode: If ``True``, return raw envelope values. Defaults to ``True``.
        :type value_mode: bool, optional
        :param query_mode: If ``True``, return the load case coordinate at the envelope extrema. Defaults to ``False``.
        :type query_mode: bool, optional
        :param extrema: Envelope direction — either ``"min"`` or ``"max"``.
        :type extrema: str, optional
        :param elements: Specific element tags to include in the envelope.
        :type elements: list, optional
        :param nodes: Specific node tags to include in the envelope.
        :type nodes: list, optional

        """
        self.value = True
        self.ds = ds
        if ds is None:
            return

        # instantiate variables
        self.load_effect = (
            load_effect  # array load effect either displacements or forces
        )
        self.envelope_ds = None
        # default xarray function name
        self.xarray_command = {
            "query": ["idxmax", "idxmin"],
            "minmax value": ["max", "min"],
            "index": ["argmax", "argmin"],
        }
        self.selected_xarray_command = []
        # get keyword args
        self.elements = kwargs.get(
            "elements", None
        )  # specific elements to query/envelope
        self.nodes = kwargs.get("nodes", None)  # specific nodes to query/envelope
        self.component = kwargs.get(
            "load_effect", None
        )  # specific load effect to query
        self.array = kwargs.get("array", "displacements")
        self.value_mode = kwargs.get("value_mode", True)
        self.query_mode = kwargs.get("query_mode", False)  # default query mode
        self.extrema = kwargs.get("extrema", "max")

        # check variables
        if self.load_effect is None:
            raise ValueError(
                "Missing argument for load_effect=: Hint requires a namestring of load"
                "effect type based on the Component dimension of the ospgrillage data"
                "set result format"
            )

        # process variables
        self.extrema_index = 0 if self.extrema == "max" else 1  # minima
        if self.query_mode:
            self.selected_xarray_command = self.xarray_command["query"][
                self.extrema_index
            ]
        elif self.value_mode:
            self.selected_xarray_command = self.xarray_command["minmax value"][
                self.extrema_index
            ]
        else:  # default to argmax/ argmin
            self.selected_xarray_command = self.xarray_command["index"][
                self.extrema_index
            ]

        # NOTE: element/component filtering via self.elements and self.component
        # is not yet implemented — get() currently returns the full reduced array.
        # A future enhancement should build a sel() dict from these kwargs and
        # call .sel(**sel_kwargs) in get().

    def get(self):
        """

        :returns: The enveloped xarray DataArray with envelope values applied.
        :rtype: xarray.DataArray
        """
        da = getattr(self.ds, self.array)
        return getattr(da, self.selected_xarray_command)(dim="Loadcase")


class _BaseInfluence:
    """Shared implementation for influence-line and influence-surface helpers."""

    def __init__(self, ds, component: str = None, **kwargs):
        self.ds = ds
        self.component = component if component is not None else kwargs.get("load_effect", None)
        self.array = kwargs.get("array", "displacements")
        self.node = kwargs.get("node", None)
        self.element = kwargs.get("element", None)
        self.loadcase = kwargs.get("loadcase", None)
        self.values = kwargs.get("values", None)
        self.influence_line = kwargs.get("influence_line", None)
        self.influence_surface = kwargs.get("influence_surface", None)

        if ds is None:
            raise ValueError("Missing ds=: an xarray Dataset is required")
        if self.component is None:
            raise ValueError("Missing component=: specify a Component label to extract")

    def _get_base_response(self):
        da = _select_response_data(
            ds=self.ds,
            array=self.array,
            component=self.component,
            node=self.node,
            element=self.element,
            loadcase=self.loadcase,
        )
        if "InfluenceLine" in da.dims:
            if self.influence_line is None:
                raise ValueError(
                    "Dataset contains multiple InfluenceLine studies; specify "
                    "influence_line= or extract each line explicitly."
                )
            da = da.sel(InfluenceLine=self.influence_line)
        if "InfluenceSurface" in da.dims:
            if self.influence_surface is None:
                raise ValueError(
                    "Dataset contains multiple InfluenceSurface studies; specify "
                    "influence_surface= explicitly."
                )
            da = da.sel(InfluenceSurface=self.influence_surface)
        if "Loadcase" not in da.dims:
            raise ValueError(
                "Selected response does not have a Loadcase dimension. Influence queries "
                "require results for multiple loading positions."
            )
        return da, _get_position_index(da, values=self.values)


class InfluenceLine(_BaseInfluence):
    """
    Class for extracting a response history against one load-position coordinate.

    Call :func:`InfluenceLine.get` to obtain an ``xarray.DataArray`` indexed by
    the selected load-position axis.
    """

    def __init__(self, ds, component: str = None, **kwargs):
        super().__init__(ds=ds, component=component, **kwargs)
        self.load_coord = kwargs.get("load_coord", "x")
        if self.load_coord not in {"x", "y", "z", "station"}:
            raise ValueError("load_coord must be one of 'x', 'y', 'z', or 'station'")

    def get(self):
        da, position_index = self._get_base_response()
        station = _cumulative_path_station(position_index)
        da = da.assign_coords(
            x=("Loadcase", position_index.sel(position_component="x").values),
            y=("Loadcase", position_index.sel(position_component="y").values),
            z=("Loadcase", position_index.sel(position_component="z").values),
            station=("Loadcase", station),
        )
        da = da.swap_dims({"Loadcase": self.load_coord}).sortby(self.load_coord)
        return da.drop_vars("Loadcase", errors="ignore")


class InfluenceSurface(_BaseInfluence):
    """
    Class for extracting a response field across a 2D load-position grid.

    Call :func:`InfluenceSurface.get` to obtain an ``xarray.DataArray`` indexed by
    two load-position coordinates.
    """

    def __init__(self, ds, component: str = None, **kwargs):
        super().__init__(ds=ds, component=component, **kwargs)
        self.x_coord = kwargs.get("x_coord", "x")
        self.y_coord = kwargs.get("y_coord", "z")
        valid_coords = {"x", "y", "z", "longitudinal_station", "transverse_station"}
        if self.x_coord not in valid_coords or self.y_coord not in valid_coords:
            raise ValueError(
                "x_coord and y_coord must each be one of 'x', 'y', 'z', "
                "'longitudinal_station', or 'transverse_station'"
            )
        if self.x_coord == self.y_coord:
            raise ValueError("x_coord and y_coord must be different coordinates")

    def get(self):
        da, position_index = self._get_base_response()
        da = da.assign_coords(
            x=("Loadcase", position_index.sel(position_component="x").values),
            y=("Loadcase", position_index.sel(position_component="y").values),
            z=("Loadcase", position_index.sel(position_component="z").values),
        )
        if "load_position_longitudinal_station" in da.coords:
            da = da.assign_coords(
                longitudinal_station=(
                    "Loadcase",
                    da.coords["load_position_longitudinal_station"].values,
                )
            )
        if "load_position_transverse_station" in da.coords:
            da = da.assign_coords(
                transverse_station=(
                    "Loadcase",
                    da.coords["load_position_transverse_station"].values,
                )
            )
        if da.indexes.get("Loadcase", None) is not None and not da.indexes["Loadcase"].is_unique:
            raise ValueError("Loadcase coordinates must be unique before building an influence surface")
        da = da.set_index(Loadcase=[self.x_coord, self.y_coord]).unstack("Loadcase")
        return da.sortby(self.x_coord).sortby(self.y_coord)


def _normalise_influence_line_input(il):
    """Return ``[(label, data_array), ...]`` for one or more influence lines."""
    if isinstance(il, xr.DataArray):
        label = il.name or "Influence Line"
        return [(label, il)]
    if isinstance(il, dict):
        return [(str(label), da) for label, da in il.items()]
    if isinstance(il, (list, tuple)):
        lines = []
        for idx, da in enumerate(il, start=1):
            label = getattr(da, "name", None) or f"Influence Line {idx}"
            lines.append((label, da))
        return lines
    raise TypeError("plot_il expects a DataArray, list/tuple of DataArrays, or dict of labelled DataArrays")


def _plot_il_path_plotly(lines, **kwargs):
    """Plot one or more influence lines along their load paths on the bridge model."""
    go = _import_plotly()
    dataset = kwargs.get("dataset", None)
    if dataset is None:
        raise ValueError("plot_il(view='path') requires dataset= with model geometry metadata")

    proxy = model_proxy_from_results(dataset)
    fig = kwargs.get("ax", None)
    if fig is None:
        fig = plot_model(
            proxy,
            backend="plotly",
            show=False,
            title=kwargs.get("title", None),
            members=kwargs.get("members", None),
            show_nodes=kwargs.get("show_nodes", True),
            show_node_labels=kwargs.get("show_node_labels", False),
            show_supports=kwargs.get("show_supports", True),
            show_rigid_links=kwargs.get("show_rigid_links", True),
        )

    scale = kwargs.get("scale", 1.0)
    fill = kwargs.get("fill", True)
    fill_alpha = kwargs.get("fill_alpha", 0.35)
    marker = kwargs.get("marker", "circle")
    show_top_markers = kwargs.get("show_top_markers", False)
    show_path_markers = kwargs.get("show_path_markers", True)
    color = kwargs.get("color", None)
    ordinate_aspect = float(kwargs.get("ordinate_aspect", 0.8))
    default_colours = ["black", "blue", "red", "green", "orange", "purple", "brown", "grey"]

    def _build_segmented_ribbon_mesh(x_vals, z_vals, base_vals, top_vals, eps=1e-12):
        """Build a non-self-intersecting ribbon mesh between top and baseline.

        The mesh is built per segment and explicitly split at ordinate
        sign-changes (top crossing baseline) to avoid folded quads.
        """
        vertices = []
        i_idx, j_idx, k_idx = [], [], []

        def add_triangle(p0, p1, p2):
            start = len(vertices)
            vertices.extend([p0, p1, p2])
            i_idx.append(start)
            j_idx.append(start + 1)
            k_idx.append(start + 2)

        for seg_idx in range(len(x_vals) - 1):
            x0, x1 = float(x_vals[seg_idx]), float(x_vals[seg_idx + 1])
            y0, y1 = float(z_vals[seg_idx]), float(z_vals[seg_idx + 1])
            b0, b1 = float(base_vals[seg_idx]), float(base_vals[seg_idx + 1])
            t0, t1 = float(top_vals[seg_idx]), float(top_vals[seg_idx + 1])
            o0 = t0 - b0
            o1 = t1 - b1

            if not np.isfinite([x0, x1, y0, y1, b0, b1, t0, t1]).all():
                continue

            if abs(o0) <= eps:
                o0 = 0.0
                t0 = b0
            if abs(o1) <= eps:
                o1 = 0.0
                t1 = b1
            if o0 == 0.0 and o1 == 0.0:
                continue

            p0_top = (x0, y0, t0)
            p1_top = (x1, y1, t1)
            p0_base = (x0, y0, b0)
            p1_base = (x1, y1, b1)

            if o0 == 0.0:
                # Segment starts on baseline: one wedge triangle.
                add_triangle(p0_base, p1_top, p1_base)
                continue
            if o1 == 0.0:
                # Segment ends on baseline: one wedge triangle.
                add_triangle(p0_top, p1_base, p0_base)
                continue

            if o0 * o1 > 0.0:
                # Same sign: one quad = two triangles.
                add_triangle(p0_top, p1_top, p0_base)
                add_triangle(p1_top, p1_base, p0_base)
                continue

            # Ordinates change sign -> split at zero crossing.
            alpha = abs(o0) / (abs(o0) + abs(o1))
            xc = x0 + alpha * (x1 - x0)
            yc = y0 + alpha * (y1 - y0)
            bc = b0 + alpha * (b1 - b0)
            p_cross = (xc, yc, bc)  # top == base at crossing

            add_triangle(p0_top, p_cross, p0_base)
            add_triangle(p1_top, p1_base, p_cross)

        if not vertices:
            return None
        vertices = np.asarray(vertices, dtype=float)
        return {
            "x": vertices[:, 0],
            "y": vertices[:, 1],
            "z": vertices[:, 2],
            "i": np.asarray(i_idx, dtype=int),
            "j": np.asarray(j_idx, dtype=int),
            "k": np.asarray(k_idx, dtype=int),
        }

    def _iter_valid_runs(mask):
        start = None
        for idx, is_valid in enumerate(mask):
            if is_valid and start is None:
                start = idx
            elif not is_valid and start is not None:
                if idx - start >= 2:
                    yield start, idx
                start = None
        if start is not None and len(mask) - start >= 2:
            yield start, len(mask)

    for idx, (label, da) in enumerate(lines):
        x_vals = np.asarray(da.coords["x"].values, dtype=float)
        y_vals = (
            np.asarray(da.coords["y"].values, dtype=float)
            if "y" in da.coords
            else np.zeros_like(x_vals, dtype=float)
        )
        z_vals = np.asarray(da.coords["z"].values, dtype=float)
        ord_vals = np.asarray(da.values, dtype=float) * scale
        base_vals = -y_vals
        top_vals = base_vals + ord_vals
        valid = (
            np.isfinite(x_vals)
            & np.isfinite(z_vals)
            & np.isfinite(base_vals)
            & np.isfinite(top_vals)
        )
        if np.count_nonzero(valid) < 2:
            continue

        x_plot = x_vals.copy()
        z_plot = z_vals.copy()
        base_plot = base_vals.copy()
        top_plot = top_vals.copy()
        x_plot[~valid] = np.nan
        z_plot[~valid] = np.nan
        base_plot[~valid] = np.nan
        top_plot[~valid] = np.nan

        if color is not None and not isinstance(color, (list, tuple)):
            line_colour = color
        elif isinstance(color, (list, tuple)):
            line_colour = color[idx]
        else:
            line_colour = default_colours[idx % len(default_colours)]
        line_dict = None
        if color is not None and not isinstance(color, (list, tuple)):
            line_dict = dict(color=line_colour, width=6)
        elif isinstance(color, (list, tuple)):
            line_dict = dict(color=line_colour, width=6)
        else:
            line_dict = dict(color=line_colour, width=6)
        top_mode = "lines+markers" if show_top_markers else "lines"
        top_marker_dict = dict(symbol=marker, size=4) if show_top_markers and marker else None

        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=z_plot,
                z=top_plot,
                mode=top_mode,
                line=line_dict,
                marker=top_marker_dict,
                name=label,
            )
        )
        base_mode = "lines+markers" if show_path_markers else "lines"
        base_marker = dict(symbol=marker, size=4) if show_path_markers and marker else None
        fig.add_trace(
            go.Scatter3d(
                x=x_plot,
                y=z_plot,
                z=base_plot,
                mode=base_mode,
                line=dict(
                    color=line_colour,
                    width=3,
                ),
                marker=base_marker,
                name=f"{label} baseline",
                showlegend=False,
            )
        )
        if fill:
            finite_ord = np.abs(top_vals[valid] - base_vals[valid])
            ord_span = float(np.nanmax(finite_ord)) if finite_ord.size else 0.0
            zero_tol = float(kwargs.get("zero_tol", max(1e-10, ord_span * 1e-12)))
            for run_start, run_end in _iter_valid_runs(valid):
                ribbon = _build_segmented_ribbon_mesh(
                    x_vals=x_vals[run_start:run_end],
                    z_vals=z_vals[run_start:run_end],
                    base_vals=base_vals[run_start:run_end],
                    top_vals=top_vals[run_start:run_end],
                    eps=zero_tol,
                )
                if ribbon is None:
                    continue
                fig.add_trace(
                    go.Mesh3d(
                        x=ribbon["x"],
                        y=ribbon["y"],
                        z=ribbon["z"],
                        i=ribbon["i"],
                        j=ribbon["j"],
                        k=ribbon["k"],
                        color=line_colour,
                        opacity=fill_alpha,
                        flatshading=True,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    aspect = _spatial_aspect_ratio(fig)
    aspect["z"] = ordinate_aspect
    fig.update_layout(
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="z (m)",
            zaxis_title=kwargs.get("ylabel", "ordinate"),
            aspectmode="manual",
            aspectratio=aspect,
        )
    )
    return fig


def plot_il(il, **kwargs):
    """Plot a 1D influence line DataArray.

    Parameters
    ----------
    il : xarray.DataArray
        Influence line with exactly one dimension.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.
    show : bool, default True
        Whether to display the figure immediately.
    ax : matplotlib.axes.Axes or plotly.graph_objects.Figure, optional
        Existing target to draw into.
    title, xlabel, ylabel, color, marker : optional
        Standard plotting customisation keywords.
    """
    backend = kwargs.get("backend", "matplotlib")
    show = kwargs.get("show", True)
    title = kwargs.get("title", None)
    xlabel = kwargs.get("xlabel", None)
    ylabel = kwargs.get("ylabel", "ordinate")
    color = kwargs.get("color", None)
    marker = kwargs.get("marker", None)
    view = kwargs.get("view", "ordinate")
    lines = _normalise_influence_line_input(il)
    for _, da in lines:
        if len(da.dims) != 1:
            raise ValueError("plot_il requires 1D DataArray inputs")

    x_name = lines[0][1].dims[0]
    if xlabel is None:
        xlabel = x_name

    if view == "path":
        if backend != "plotly":
            raise ValueError("plot_il(view='path') currently requires backend='plotly'")
        fig = _plot_il_path_plotly(lines, **kwargs)
        if title is not None:
            fig.update_layout(title=title)
        if show:
            fig.show()
        return fig

    if backend == "matplotlib":
        ax = kwargs.get("ax", None)
        if ax is None:
            figsize = kwargs.get("figsize", None)
            _, ax = plt.subplots(figsize=figsize)
        for idx, (label, da) in enumerate(lines):
            line_kwargs = {}
            if color is not None and not isinstance(color, (list, tuple)):
                line_kwargs["color"] = color
            elif isinstance(color, (list, tuple)):
                line_kwargs["color"] = color[idx]
            if marker is not None:
                line_kwargs["marker"] = marker
            ax.plot(
                da.coords[da.dims[0]].values,
                da.values,
                label=label,
                **line_kwargs,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if len(lines) > 1:
            ax.legend()
        if show:
            plt.show()
        return ax

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = kwargs.get("ax", None)
        if fig is None:
            fig = go.Figure()
        for idx, (label, da) in enumerate(lines):
            line_dict = None
            if color is not None and not isinstance(color, (list, tuple)):
                line_dict = dict(color=color)
            elif isinstance(color, (list, tuple)):
                line_dict = dict(color=color[idx])
            marker_dict = dict(symbol=marker) if marker else None
            fig.add_trace(
                go.Scatter(
                    x=da.coords[da.dims[0]].values,
                    y=da.values,
                    mode="lines+markers" if marker else "lines",
                    line=line_dict,
                    marker=marker_dict,
                    name=label,
                )
            )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
        )
        if show:
            fig.show()
        return fig

    raise ValueError("Unknown backend: choose 'matplotlib' or 'plotly'")


def _grid_triangles_from_mask(valid_mask):
    """Build triangle connectivity from a structured grid validity mask."""
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be a 2D array")

    rows, cols = valid_mask.shape
    compact_index = -np.ones_like(valid_mask, dtype=int)
    compact_index[valid_mask] = np.arange(np.count_nonzero(valid_mask))

    triangles = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            v00 = (row, col)
            v01 = (row, col + 1)
            v11 = (row + 1, col + 1)
            v10 = (row + 1, col)

            if valid_mask[v00] and valid_mask[v01] and valid_mask[v11]:
                triangles.append(
                    [
                        compact_index[v00],
                        compact_index[v01],
                        compact_index[v11],
                    ]
                )
            if valid_mask[v00] and valid_mask[v11] and valid_mask[v10]:
                triangles.append(
                    [
                        compact_index[v00],
                        compact_index[v11],
                        compact_index[v10],
                    ]
                )

    if triangles:
        return np.asarray(triangles, dtype=int)
    return np.empty((0, 3), dtype=int)


def _prepare_influence_surface_plot_data(isurface, coordinate_space):
    """Prepare coordinate arrays and optional triangulation metadata for ``plot_is``."""
    x_name, y_name = isurface.dims
    ordinate = np.asarray(isurface.values.T, dtype=float)

    use_physical_coords = (
        coordinate_space in {"auto", "physical"}
        and "x" in isurface.coords
        and "z" in isurface.coords
        and isurface.coords["x"].dims == isurface.dims
        and isurface.coords["z"].dims == isurface.dims
    )
    if coordinate_space == "station":
        use_physical_coords = False

    if use_physical_coords:
        x_grid = np.asarray(isurface.coords["x"].values.T, dtype=float)
        y_grid = np.asarray(isurface.coords["z"].values.T, dtype=float)
        xlabel = "x"
        ylabel = "z"
    else:
        x_axis = np.asarray(isurface.coords[x_name].values, dtype=float)
        y_axis = np.asarray(isurface.coords[y_name].values, dtype=float)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        xlabel = x_name
        ylabel = y_name

    valid_mask = (
        np.isfinite(x_grid)
        & np.isfinite(y_grid)
        & np.isfinite(ordinate)
    )
    use_triangulation = bool(use_physical_coords or np.any(~valid_mask))

    tri_data = None
    if use_triangulation:
        triangles = _grid_triangles_from_mask(valid_mask)
        x_points = x_grid[valid_mask]
        y_points = y_grid[valid_mask]
        ordinates = ordinate[valid_mask]

        if len(x_points) < 3 or len(triangles) == 0:
            raise ValueError(
                "Influence surface plotting requires at least three connected "
                "valid points when using physical coordinates"
            )
        tri_data = {
            "x": x_points,
            "y": y_points,
            "z": ordinates,
            "triangles": triangles,
        }

    return {
        "x_name": x_name,
        "y_name": y_name,
        "ordinate": ordinate,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "use_triangulation": use_triangulation,
        "tri_data": tri_data,
    }


def plot_is(isurface, **kwargs):
    """Plot a 2D influence surface DataArray.

    Parameters
    ----------
    isurface : xarray.DataArray
        Influence surface with exactly two dimensions.
    backend : {"matplotlib", "plotly"}, default "matplotlib"
        Rendering backend.
    show : bool, default True
        Whether to display the figure immediately.
    ax : matplotlib.axes.Axes or plotly.graph_objects.Figure, optional
        Existing target to draw into.
    title, xlabel, ylabel, colorscale : optional
        Standard plotting customisation keywords.
    """
    if len(isurface.dims) != 2:
        raise ValueError("plot_is requires a 2D DataArray")

    backend = kwargs.get("backend", "matplotlib")
    show = kwargs.get("show", True)
    title = kwargs.get("title", None)
    colorscale = kwargs.get("colorscale", "RdBu_r")
    view = kwargs.get("view", "contour")
    coordinate_space = kwargs.get("coordinate_space", "auto")
    contour_levels = kwargs.get("levels", 20)
    opacity = kwargs.get("opacity", None)
    ordinate_aspect = float(kwargs.get("ordinate_aspect", 0.8))

    plot_data = _prepare_influence_surface_plot_data(isurface, coordinate_space)
    z = plot_data["ordinate"]
    x_grid = plot_data["x_grid"]
    y_grid = plot_data["y_grid"]
    use_triangulation = plot_data["use_triangulation"]
    tri_data = plot_data["tri_data"]
    xlabel = kwargs.get("xlabel", plot_data["xlabel"])
    ylabel = kwargs.get("ylabel", plot_data["ylabel"])

    if backend == "matplotlib":
        ax = kwargs.get("ax", None)
        if ax is None:
            figsize = kwargs.get("figsize", None)
            if view == "surface3d":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")
            else:
                _, ax = plt.subplots(figsize=figsize)
        if view == "surface3d":
            if use_triangulation:
                surface = ax.plot_trisurf(
                    tri_data["x"],
                    tri_data["y"],
                    tri_data["z"],
                    triangles=tri_data["triangles"],
                    cmap=colorscale,
                )
            else:
                surface = ax.plot_surface(x_grid, y_grid, z, cmap=colorscale)
            ax.set_zlabel("ordinate")
            plt.colorbar(surface, ax=ax, label="ordinate")
        else:
            if use_triangulation:
                import matplotlib.tri as mtri

                tri = mtri.Triangulation(
                    tri_data["x"],
                    tri_data["y"],
                    tri_data["triangles"],
                )
                contour = ax.tricontourf(
                    tri,
                    tri_data["z"],
                    levels=contour_levels,
                    cmap=colorscale,
                )
            else:
                contour = ax.contourf(x_grid, y_grid, z, levels=contour_levels, cmap=colorscale)
            plt.colorbar(contour, ax=ax, label="ordinate")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
        return ax

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = kwargs.get("ax", None)
        has_existing_traces = fig is not None and len(getattr(fig, "data", ())) > 0
        surface_hover = bool(kwargs.get("surface_hover", not has_existing_traces))
        if fig is None:
            fig = go.Figure()
        colorbar_x = kwargs.get("colorbar_x", -0.08 if has_existing_traces else 1.02)
        colorbar_kw = dict(title="ordinate", x=colorbar_x, xanchor="right")
        if view == "surface3d":
            if use_triangulation:
                fig.add_trace(
                    go.Mesh3d(
                        x=tri_data["x"],
                        y=tri_data["y"],
                        z=tri_data["z"],
                        i=tri_data["triangles"][:, 0],
                        j=tri_data["triangles"][:, 1],
                        k=tri_data["triangles"][:, 2],
                        intensity=tri_data["z"],
                        colorscale=colorscale,
                        colorbar=colorbar_kw,
                        showscale=True,
                        name="Influence Surface",
                        hovertemplate=(
                            f"{xlabel}: %{{x:.3f}}<br>"
                            f"{ylabel}: %{{y:.3f}}<br>"
                            "ordinate: %{intensity:.3g}<extra></extra>"
                        ) if surface_hover else None,
                        hoverinfo="skip" if not surface_hover else None,
                    )
                )
            else:
                fig.add_trace(
                    go.Surface(
                        x=x_grid,
                        y=y_grid,
                        z=z,
                        colorscale=colorscale,
                        colorbar=colorbar_kw,
                        hoverinfo="skip" if not surface_hover else None,
                    )
                )
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    zaxis_title="ordinate",
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=ordinate_aspect),
                ),
            )
        else:
            if use_triangulation:
                mesh_kwargs = {}
                if opacity is not None:
                    mesh_kwargs["opacity"] = opacity
                fig.add_trace(
                    go.Mesh3d(
                        x=tri_data["x"],
                        y=tri_data["y"],
                        z=np.zeros_like(tri_data["z"]),
                        i=tri_data["triangles"][:, 0],
                        j=tri_data["triangles"][:, 1],
                        k=tri_data["triangles"][:, 2],
                        intensity=tri_data["z"],
                        colorscale=colorscale,
                        colorbar=colorbar_kw,
                        showscale=True,
                        name="Influence Surface",
                        hovertemplate=(
                            f"{xlabel}: %{{x:.3f}}<br>"
                            f"{ylabel}: %{{y:.3f}}<br>"
                            "ordinate: %{intensity:.3g}<extra></extra>"
                        ) if surface_hover else None,
                        hoverinfo="skip" if not surface_hover else None,
                        **mesh_kwargs,
                    )
                )
                fig.update_layout(
                    title=title,
                    scene=dict(
                        xaxis_title=xlabel,
                        yaxis_title=ylabel,
                        zaxis=dict(
                            title="",
                            showticklabels=False,
                            visible=False,
                        ),
                        aspectmode="data",
                        camera=dict(
                            projection=dict(type="orthographic"),
                            eye=dict(x=0.0, y=0.0, z=2.2),
                            up=dict(x=0, y=1, z=0),
                        ),
                    ),
                )
                aspect = _spatial_aspect_ratio(fig)
                aspect["z"] = ordinate_aspect
                fig.update_layout(
                    scene=dict(
                        aspectmode="manual",
                        aspectratio=aspect,
                    )
                )
            else:
                fig.add_trace(
                    go.Heatmap(
                        x=x_grid[0, :],
                        y=y_grid[:, 0],
                        z=z,
                        colorscale=colorscale,
                        colorbar=colorbar_kw,
                        hoverinfo="skip" if not surface_hover else None,
                    )
                )
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                )
        if show:
            fig.show()
        return fig

    raise ValueError("Unknown backend: choose 'matplotlib' or 'plotly'")


# ---------------------------------------------------------------------------
# Force component lookup tables
# ---------------------------------------------------------------------------
_FORCE_COMP_DICT = {"Fx": 0, "Fy": 1, "Fz": 2, "Mx": 3, "My": 4, "Mz": 5}
_FORCE_COMP_FACTOR = {"Fx": 1, "Fy": 1, "Fz": 1, "Mx": 1, "My": 1, "Mz": -1}

_FORCE_COMPONENTS = [
    "Vx_i",
    "Vy_i",
    "Vz_i",
    "Mx_i",
    "My_i",
    "Mz_i",
    "Vx_j",
    "Vy_j",
    "Vz_j",
    "Mx_j",
    "My_j",
    "Mz_j",
]

# Shell contour component lookup.  Each shell element has 24 force
# components (6 per node x 4 nodes i/j/k/l).  The mapping below
# resolves a user-facing component name to the 4 column names in
# ``forces_shell`` aligned with the ``ele_nodes_shell`` Nodes dimension.
_SHELL_COMPONENTS = ("Vx", "Vy", "Vz", "Mx", "My", "Mz")
_SHELL_COMP_COLUMNS = {
    comp: [f"{comp}_{s}" for s in ("i", "j", "k", "l")]
    for comp in _SHELL_COMPONENTS
}

# Displacement components for shell contour.  User-facing names map to
# the Component coordinate in the ``displacements`` DataArray.
_DISP_COMPONENTS = {"Dx": "x", "Dy": "y", "Dz": "z"}

# Shell section stress resultants (from Gauss-point data in stresses_shell).
# User-facing name → list of 4 GP column names in the ``Stress`` dimension.
_STRESS_RESULTANTS = ("N11", "N22", "N12", "M11", "M22", "M12", "Q13", "Q23")
_STRESS_COMP_COLUMNS = {
    sr: [f"{sr}_{gp}" for gp in ("gp1", "gp2", "gp3", "gp4")]
    for sr in _STRESS_RESULTANTS
}

# All components accepted by plot_srf
_SRF_COMPONENTS = (
    _SHELL_COMPONENTS
    + tuple(_DISP_COMPONENTS.keys())
    + _STRESS_RESULTANTS
)


# ---------------------------------------------------------------------------
# Data extraction helpers (backend-agnostic)
# ---------------------------------------------------------------------------
def _num_z_groups(ospgrillage_obj, member):
    """Return the number of z-groups for a member.

    Longitudinal members like ``"interior_main_beam"`` may contain several
    parallel beams.  Each is a separate z-group.  Transverse and edge
    member types always return 1.
    """
    z_grp = ospgrillage_obj.common_grillage_element_z_group.get(member)
    if z_grp is None:
        return 1
    return len(z_grp)


def _extract_force_data(
    ospgrillage_obj,
    result_obj,
    component,
    member,
    loadcase=None,
    option="elements",
    z_group_num=0,
):
    """Extract force distribution data for a single member.

    Iterates over every element in the requested *member* group, calls
    :func:`opsvis.section_force_distribution_3d` for each, and collects the
    results into three parallel lists.

    :param ospgrillage_obj: Grillage model object.
    :type ospgrillage_obj: OspGrillage
    :param result_obj: xarray DataSet of analysis results.
    :param component: Force component name (``"Fx"``, ``"Fy"``, ``"Fz"``,
        ``"Mx"``, ``"My"``, ``"Mz"``) or integer index.
    :param member: Member group name string.
    :param loadcase: Load case name.  ``None`` uses the first load case.
    :param option: Element query option passed to
        :meth:`~ospgrillage.osp_grillage.OspGrillage.get_element`.
    :returns: ``(all_xx, all_zz, all_values)`` — each a list of
        per-element :class:`numpy.ndarray`.  ``all_values`` entries are
        already multiplied by the component sign factor.
    :rtype: tuple[list, list, list]
    """
    comp_index = (
        component if isinstance(component, int) else _FORCE_COMP_DICT[component]
    )
    factor = _FORCE_COMP_FACTOR[component] if isinstance(component, str) else 1

    nodes = ospgrillage_obj.get_nodes()
    eletag = ospgrillage_obj.get_element(
        member=member,
        options=option,
        z_group_num=z_group_num,
    )

    if ospgrillage_obj.model_type == "shell_beam":
        force_result = result_obj.forces_beam
    else:
        force_result = result_obj.forces

    all_xx, all_zz, all_values = [], [], []

    for ele in eletag:
        # get force components for this element
        if loadcase:
            ele_components = force_result.sel(
                Loadcase=loadcase,
                Element=ele,
                Component=_FORCE_COMPONENTS,
            ).values
        else:
            ele_components = force_result.sel(
                Element=ele,
                Component=_FORCE_COMPONENTS,
            )[0].values

        # get nodes of element
        if ospgrillage_obj.model_type == "shell_beam":
            ele_node = result_obj.ele_nodes_shell.sel(Element=ele)
            if all(np.isnan(result_obj.ele_nodes_shell.sel(Element=ele))):
                ele_node = result_obj.ele_nodes_beam.sel(Element=ele)
            ele_node = ele_node[~np.isnan(ele_node)]
        else:
            ele_node = result_obj.ele_nodes.sel(Element=ele)

        node_tags = ele_node.values.flatten().astype(int).tolist()
        xx = [nodes[n]["coordinate"][0] for n in node_tags]
        yy = [nodes[n]["coordinate"][1] for n in node_tags]
        zz = [nodes[n]["coordinate"][2] for n in node_tags]

        # opsvis section force distribution
        s, *_ = opsv.section_force_distribution_3d(
            ecrd=np.column_stack([xx, yy, zz]),
            pl=ele_components,
        )

        all_xx.append(np.array(xx))
        all_zz.append(np.array(zz))
        all_values.append(s[:, comp_index] * factor)

    return all_xx, all_zz, all_values


def _extract_def_data(
    ospgrillage_obj,
    result_obj,
    member,
    component="y",
    loadcase=None,
    option="nodes",
    z_group_num=0,
):
    """Extract displacement data for a single member.

    Iterates over the nodes belonging to *member* and collects their
    coordinates and displacement values.

    :param ospgrillage_obj: Grillage model object.
    :type ospgrillage_obj: OspGrillage
    :param result_obj: xarray DataSet of analysis results.
    :param member: Member group name string.
    :param component: Displacement component (``"x"``, ``"y"``, ``"z"``).
        Defaults to ``"y"`` (vertical deflection).
    :param loadcase: Load case name.  ``None`` uses the first load case.
    :param option: Node query option passed to
        :meth:`~ospgrillage.osp_grillage.OspGrillage.get_element`.
    :returns: ``(xx_list, zz_list, values_list)`` — node x-coordinates,
        z-coordinates, and displacement scalars.
    :rtype: tuple[list, list, list]
    """
    plot_option = option if option is not None else "nodes"
    dis_comp = component if component is not None else "y"

    nodes = ospgrillage_obj.get_nodes()
    node_result = ospgrillage_obj.get_element(
        member=member,
        options=plot_option,
        z_group_num=z_group_num,
    )
    # Longitudinal members return [[node_list]], others return [node_list]
    nodes_to_plot = (
        node_result[0]
        if node_result and isinstance(node_result[0], list)
        else node_result
    )

    xx_list, zz_list, values_list = [], [], []

    for node in nodes_to_plot:
        if loadcase:
            disp = result_obj.displacements.sel(
                Component=dis_comp,
                Node=node,
                Loadcase=loadcase,
            ).values
        else:
            disp = result_obj.displacements.sel(
                Component=dis_comp,
                Node=node,
            )[0].values
        xx_list.append(nodes[node]["coordinate"][0])
        zz_list.append(nodes[node]["coordinate"][2])
        values_list.append(float(disp))

    return xx_list, zz_list, values_list


# ---------------------------------------------------------------------------
# Shell contour data extraction
# ---------------------------------------------------------------------------
def _extract_shell_contour_data(result_obj, component, loadcase=None, *, averaging="nodal"):
    """Extract per-node contour values for a shell stress resultant.

    Parameters
    ----------
    result_obj : xarray.Dataset
        Results dataset containing ``forces_shell`` and ``ele_nodes_shell``.
    component : str
        One of ``"Vx"``, ``"Vy"``, ``"Vz"``, ``"Mx"``, ``"My"``, ``"Mz"``.
    loadcase : str or None
        Load case name.  ``None`` uses the first load case.
    averaging : str
        ``"nodal"`` (default) — average contributions from all elements
        sharing a node.  Future: ``"none"`` for raw per-element values.

    Returns
    -------
    node_values : dict[int, float]
        ``{node_tag: averaged_value}``
    element_quads : list[tuple[int, ...]]
        Each entry is the node tags of a shell element (3 or 4 nodes).
    """
    if component not in _SHELL_COMPONENTS:
        raise ValueError(
            f"Unknown shell component {component!r}. "
            f"Expected one of {_SHELL_COMPONENTS}."
        )

    columns = _SHELL_COMP_COLUMNS[component]
    forces = result_obj["forces_shell"]
    ele_nodes = result_obj["ele_nodes_shell"]

    # Select loadcase
    if loadcase is not None:
        forces = forces.sel(Loadcase=loadcase)
    else:
        forces = forces.isel(Loadcase=0)

    # Build element connectivity and per-node accumulator
    from collections import defaultdict

    accum = defaultdict(list)  # node_tag -> [value, ...]
    element_quads = []

    for ele in ele_nodes.coords["Element"].values:
        tags_raw = ele_nodes.sel(Element=ele).values.flatten()
        tags = [int(t) for t in tags_raw if not np.isnan(t)]
        if len(tags) < 3:
            continue
        element_quads.append(tuple(tags))

        # Extract the component value at each node of this element
        col_subset = columns[: len(tags)]
        vals = forces.sel(Element=ele, Component=col_subset).values.flatten()
        for tag, val in zip(tags, vals):
            accum[tag].append(float(val))

    if averaging == "nodal":
        node_values = {tag: np.mean(vals) for tag, vals in accum.items()}
    else:
        # Fallback: nodal averaging (extensible for "none", "centroid")
        node_values = {tag: np.mean(vals) for tag, vals in accum.items()}

    return node_values, element_quads


def _extract_shell_disp_data(result_obj, component, loadcase=None):
    """Extract per-node displacement values over the shell mesh.

    Unlike force data, displacements are already per-node so no
    averaging is needed — just filter to the nodes that belong to
    shell elements.

    Parameters
    ----------
    result_obj : xarray.Dataset
    component : str
        One of ``"Dx"``, ``"Dy"``, ``"Dz"``.
    loadcase : str or None

    Returns
    -------
    node_values : dict[int, float]
    element_quads : list[tuple[int, ...]]
    """
    ds_comp = _DISP_COMPONENTS[component]
    disps = result_obj["displacements"]
    ele_nodes = result_obj["ele_nodes_shell"]

    # Select loadcase
    if loadcase is not None:
        disps = disps.sel(Loadcase=loadcase)
    else:
        disps = disps.isel(Loadcase=0)

    # Build element connectivity (same as force extraction)
    element_quads = []
    shell_node_set = set()
    for ele in ele_nodes.coords["Element"].values:
        tags_raw = ele_nodes.sel(Element=ele).values.flatten()
        tags = [int(t) for t in tags_raw if not np.isnan(t)]
        if len(tags) < 3:
            continue
        element_quads.append(tuple(tags))
        shell_node_set.update(tags)

    # Extract displacement at each shell node
    node_values = {}
    for tag in shell_node_set:
        val = float(disps.sel(Node=tag, Component=ds_comp).values)
        node_values[tag] = val

    return node_values, element_quads


def _extract_shell_stress_data(result_obj, component, loadcase=None):
    """Extract per-node stress resultant values over the shell mesh.

    The ``stresses_shell`` DataArray stores 8 stress resultants at 4
    Gauss points per element.  This function averages the requested
    component across the 4 Gauss points to get one value per element,
    then averages contributions from neighbouring elements at shared
    nodes (same approach as :func:`_extract_shell_contour_data`).

    Parameters
    ----------
    result_obj : xarray.Dataset
    component : str
        One of ``"N11"``, ``"N22"``, ``"N12"``, ``"M11"``, ``"M22"``,
        ``"M12"``, ``"Q13"``, ``"Q23"``.
    loadcase : str or None

    Returns
    -------
    node_values : dict[int, float]
    element_quads : list[tuple[int, ...]]
    """
    from collections import defaultdict

    columns = _STRESS_COMP_COLUMNS[component]
    stresses = result_obj["stresses_shell"]
    ele_nodes = result_obj["ele_nodes_shell"]

    # Select loadcase
    if loadcase is not None:
        stresses = stresses.sel(Loadcase=loadcase)
    else:
        stresses = stresses.isel(Loadcase=0)

    accum = defaultdict(list)
    element_quads = []

    for ele in ele_nodes.coords["Element"].values:
        tags_raw = ele_nodes.sel(Element=ele).values.flatten()
        tags = [int(t) for t in tags_raw if not np.isnan(t)]
        if len(tags) < 3:
            continue
        element_quads.append(tuple(tags))

        # Average across the 4 Gauss points for this element
        gp_vals = stresses.sel(Element=ele, Stress=columns).values.flatten()
        ele_avg = float(np.nanmean(gp_vals))

        # Assign element average to each of its nodes
        for tag in tags:
            accum[tag].append(ele_avg)

    # Average contributions from neighbouring elements at shared nodes
    node_values = {tag: float(np.mean(vals)) for tag, vals in accum.items()}
    return node_values, element_quads


def _triangulate_shell_mesh(node_coords, element_quads):
    """Build deduplicated vertex arrays and triangle indices.

    Each quad is split into two triangles.  Shared nodes get a single
    vertex so that Plotly ``Mesh3d`` ``intensity`` interpolates smoothly.

    Parameters
    ----------
    node_coords : dict[int, sequence]
        ``{node_tag: [x, y, z]}`` in model coordinates.
    element_quads : list[tuple[int, ...]]
        Shell element connectivity (3- or 4-node).

    Returns
    -------
    vx, vy, vz : list[float]
        Vertex coordinates in Plotly convention
        (model x -> plotly x, model z -> plotly y, -model y -> plotly z).
    i_idx, j_idx, k_idx : list[int]
        Triangle vertex indices.
    tag_to_vidx : dict[int, int]
        ``{node_tag: vertex_index}`` for building the intensity array.
    """
    # Collect unique nodes in a deterministic order
    seen = {}
    ordered_tags = []
    for quad in element_quads:
        for tag in quad:
            if tag not in seen:
                seen[tag] = len(ordered_tags)
                ordered_tags.append(tag)
    tag_to_vidx = seen

    # Vertex arrays in Plotly convention
    vx, vy, vz = [], [], []
    for tag in ordered_tags:
        c = node_coords[tag]
        vx.append(c[0])       # model x -> plotly x
        vy.append(c[2])       # model z -> plotly y
        vz.append(-c[1])      # model y -> plotly z (negated)

    # Triangle indices (each quad -> 2 triangles)
    i_idx, j_idx, k_idx = [], [], []
    for quad in element_quads:
        idxs = [tag_to_vidx[t] for t in quad]
        if len(idxs) == 4:
            # Quad: (0,1,2) and (0,2,3)
            i_idx.extend([idxs[0], idxs[0]])
            j_idx.extend([idxs[1], idxs[2]])
            k_idx.extend([idxs[2], idxs[3]])
        elif len(idxs) == 3:
            i_idx.append(idxs[0])
            j_idx.append(idxs[1])
            k_idx.append(idxs[2])

    return vx, vy, vz, i_idx, j_idx, k_idx, tag_to_vidx


# ---------------------------------------------------------------------------
# Plotly 3D figure builders
# ---------------------------------------------------------------------------
def _spatial_aspect_ratio(fig):
    """Compute a Plotly aspectratio that keeps the x–y (plan) axes
    proportional while letting the z (value) axis scale freely.

    Scans all ``Scatter3d`` traces already added to *fig* and returns a
    ``dict(x=…, y=…, z=1)`` where x and y reflect the true span-to-width
    ratio of the bridge, and z is normalised to 1 so Plotly auto-scales
    the value axis to fill the available space.
    """
    all_x, all_y = [], []
    for trace in fig.data:
        if hasattr(trace, "x") and trace.x is not None:
            all_x.extend([v for v in trace.x if v is not None])
        if hasattr(trace, "y") and trace.y is not None:
            all_y.extend([v for v in trace.y if v is not None])

    if not all_x or not all_y:
        return dict(x=1, y=1, z=1)

    dx = max(all_x) - min(all_x)
    dy = max(all_y) - min(all_y)

    if dx == 0 and dy == 0:
        return dict(x=1, y=1, z=1)

    # Normalise so the larger spatial dimension = 1
    dmax = max(dx, dy, 1e-12)
    return dict(x=dx / dmax, y=dy / dmax, z=1)


def _plotly_3d_force(
    ospgrillage_obj,
    result_obj,
    component,
    members,
    loadcase=None,
    comp_label=None,
    *,
    fig=None,
    figsize=None,
    scale=1.0,
    title=_AUTO,
    alpha=1.0,
    fill=True,
    fill_alpha=None,
    show_supports=True,
):
    """Build an interactive 3D Plotly figure of force diagrams.

    Each member is drawn as a ``Scatter3d`` trace with:

    * x = longitudinal position (m)
    * y = transverse position (z-coordinate from the grillage model, m)
    * z = force component value

    A semi-transparent ``Mesh3d`` ribbon fills between the diagram curve
    and the zero baseline.  Set *fill=False* to suppress the fill.

    :param ospgrillage_obj: Grillage model object.
    :param result_obj: xarray DataSet of analysis results.
    :param component: Force component name (e.g. ``"Mz"``, ``"Fy"``).
    :param members: List of member group name strings to include.
    :param loadcase: Load case name.  ``None`` uses the first load case.
    :param comp_label: Axis / title label.  Defaults to *component*.
    :param fig: Existing Plotly ``Figure`` to add traces to.
    :param figsize: Figure size in inches ``(width, height)``.
    :param scale: Multiply plotted values by this factor.
    :param title: Plot title.  Default auto-generates from component.
        ``None`` suppresses the title.
    :param alpha: Trace opacity (0–1).
    :param fill: If ``True`` (default), draw a semi-transparent filled
        ribbon between the diagram and zero baseline.
    :param fill_alpha: Opacity of the Mesh3d fill ribbon (0–1).
        Defaults to ``alpha * 0.4`` when ``None``.
    :returns: Interactive 3D figure.
    :rtype: :class:`plotly.graph_objects.Figure`
    """
    go = _import_plotly()
    new_fig = fig is None
    if new_fig:
        fig = go.Figure()
    label = comp_label or component

    # Resolve fill opacity: user-specified fill_alpha takes precedence,
    # otherwise default to alpha * 0.5 for a clearly visible fill.
    _fill_alpha = fill_alpha if fill_alpha is not None else alpha * 0.5

    # Negate z-axis so that sagging BM and downward shear plot below
    # the baseline (structural engineering convention).
    sign = -1.0

    colours = ["black", "blue", "red", "green", "orange", "purple", "brown", "grey"]
    trace_idx = 0

    for member in members:
        n_groups = _num_z_groups(ospgrillage_obj, member)
        for zg in range(n_groups):
            try:
                all_xx, all_zz, all_vals = _extract_force_data(
                    ospgrillage_obj,
                    result_obj,
                    component,
                    member,
                    loadcase,
                    z_group_num=zg,
                )
            except (KeyError, ValueError, IndexError):
                continue  # member not available for this model type
            colour = colours[trace_idx % len(colours)]
            trace_name = member if n_groups == 1 else f"{member} [{zg+1}]"

            # Accumulate fill vertices across all elements of this member
            # so we emit ONE Mesh3d per member (renders far better than
            # many tiny per-element meshes in WebGL).
            mesh_vx, mesh_vy, mesh_vz = [], [], []
            mesh_ii, mesh_jj, mesh_kk = [], [], []
            voffset = 0

            for ex, ez, ev in zip(all_xx, all_zz, all_vals):
                sv = ev * scale * sign
                # Force diagram line
                fig.add_trace(
                    go.Scatter3d(
                        x=ex,
                        y=ez,
                        z=sv,
                        mode="lines",
                        line=dict(color=colour, width=4),
                        opacity=alpha,
                        name=trace_name,
                        showlegend=False,
                    )
                )
                # Baseline (zero) line
                fig.add_trace(
                    go.Scatter3d(
                        x=ex,
                        y=ez,
                        z=np.zeros_like(sv),
                        mode="lines",
                        line=dict(color=colour, width=1, dash="dot"),
                        opacity=alpha,
                        showlegend=False,
                    )
                )
                # Vertical drop-lines from each node to the baseline
                drop_x, drop_y, drop_z = [], [], []
                for xi, zi, vi in zip(ex, ez, sv):
                    drop_x.extend([xi, xi, None])
                    drop_y.extend([zi, zi, None])
                    drop_z.extend([vi, 0.0, None])
                fig.add_trace(
                    go.Scatter3d(
                        x=drop_x,
                        y=drop_y,
                        z=drop_z,
                        mode="lines",
                        line=dict(color=colour, width=1),
                        opacity=alpha * 0.6,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                # Collect fill vertices for the combined ribbon
                if fill:
                    n_pts = len(ex)
                    if n_pts >= 2:
                        mesh_vx.extend(list(ex) + list(ex))
                        mesh_vy.extend(list(ez) + list(ez))
                        mesh_vz.extend(list(sv) + [0.0] * n_pts)
                        for idx in range(n_pts - 1):
                            mesh_ii.extend([voffset + idx, voffset + idx + 1])
                            mesh_jj.extend(
                                [voffset + idx + 1, voffset + n_pts + idx + 1]
                            )
                            mesh_kk.extend(
                                [voffset + n_pts + idx, voffset + n_pts + idx]
                            )
                        voffset += 2 * n_pts

            # One combined Mesh3d ribbon per member
            if fill and mesh_vx:
                fig.add_trace(
                    go.Mesh3d(
                        x=mesh_vx,
                        y=mesh_vy,
                        z=mesh_vz,
                        i=mesh_ii,
                        j=mesh_jj,
                        k=mesh_kk,
                        color=colour,
                        opacity=_fill_alpha,
                        flatshading=True,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # One legend entry per beam
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="lines",
                    line=dict(color=colour, width=4),
                    name=trace_name,
                )
            )
            trace_idx += 1

    # Support markers on the zero-baseline (skip for _ModelProxy)
    if show_supports and hasattr(ospgrillage_obj, "Mesh_obj"):
        supports = _extract_support_data(ospgrillage_obj)
        from collections import defaultdict

        by_type = defaultdict(list)
        for sup in supports:
            by_type[sup["fixity_type"]].append(sup)
        for ftype, sups in sorted(by_type.items()):
            symbol, colour, sz = _SUPPORT_STYLE_PLOTLY.get(ftype, ("x", "purple", 6))
            sx = [s["x"] for s in sups]
            sy = [s["z"] for s in sups]
            sz_vals = [0.0] * len(sups)
            fig.add_trace(
                go.Scatter3d(
                    x=sx,
                    y=sy,
                    z=sz_vals,
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=sz,
                        color=colour,
                        line=dict(color="black", width=1),
                    ),
                    name=f"support ({ftype})",
                )
            )

    # Compute spatial data ranges so the plan-view (x vs z) axes are
    # proportional while the force axis scales freely.
    aspect = _spatial_aspect_ratio(fig)
    _no_bg = dict(showbackground=False)
    layout_kw = dict(
        scene=dict(
            xaxis=dict(title="x (m)", **_no_bg),
            yaxis=dict(title="z (m)", **_no_bg),
            zaxis=dict(title=label, **_no_bg),
            aspectmode="manual",
            aspectratio=aspect,
        ),
        legend_title="Member",
    )
    if new_fig:
        # Only set title when we created the figure (preserve SRF title
        # when composing onto an existing contour figure).
        if title is _AUTO:
            layout_kw["title"] = f"{label} Diagram"
        elif title is not None:
            layout_kw["title"] = title
    if figsize is not None:
        layout_kw["width"] = figsize[0] * 100
        layout_kw["height"] = figsize[1] * 100

    fig.update_layout(**layout_kw)
    return fig


def _plotly_3d_def(
    ospgrillage_obj,
    result_obj,
    members,
    component="y",
    loadcase=None,
    *,
    fig=None,
    figsize=None,
    scale=1.0,
    title=_AUTO,
    show_supports=True,
):
    """Build an interactive 3D Plotly figure of deflected shapes.

    Each member is drawn as a ``Scatter3d`` trace with markers, accompanied
    by a dotted zero-baseline.

    :param ospgrillage_obj: Grillage model object.
    :param result_obj: xarray DataSet of analysis results.
    :param members: List of member group name strings to include.
    :param component: Displacement component (default ``"y"``).
    :param loadcase: Load case name.  ``None`` uses the first load case.
    :param fig: Existing Plotly ``Figure`` to add traces to.
    :param figsize: Figure size in inches ``(width, height)``.
    :param scale: Multiply plotted values by this factor.
    :param title: Plot title.  Default auto-generates from component.
        ``None`` suppresses the title.
    :returns: Interactive 3D figure.
    :rtype: :class:`plotly.graph_objects.Figure`
    """
    go = _import_plotly()
    if fig is None:
        fig = go.Figure()

    # Negate so downward deflection plots below the baseline.
    sign = -1.0

    colours = ["blue", "red", "green", "orange", "black", "purple", "brown", "grey"]
    trace_idx = 0
    _TRANSVERSE = ("transverse_slab", "start_edge", "end_edge")

    for member in members:
        n_groups = _num_z_groups(ospgrillage_obj, member)

        if member in _TRANSVERSE:
            # Transverse members: extract per-element displacement at end
            # nodes and draw short line segments across the deck width.
            try:
                colour = colours[trace_idx % len(colours)]
                nodes_dict = ospgrillage_obj.get_nodes()
                eletags = ospgrillage_obj.get_element(
                    member=member,
                    options="elements",
                )
                if ospgrillage_obj.model_type == "shell_beam":
                    ele_node_da = result_obj.ele_nodes_beam
                else:
                    ele_node_da = result_obj.ele_nodes

                all_x, all_y, all_v = [], [], []
                for ele in eletags:
                    en = ele_node_da.sel(Element=ele).values
                    node_tags = en.flatten().astype(int).tolist()
                    for nd in node_tags:
                        c = nodes_dict[nd]["coordinate"]
                        if loadcase:
                            d = float(
                                result_obj.displacements.sel(
                                    Component=component,
                                    Node=nd,
                                    Loadcase=loadcase,
                                ).values
                            )
                        else:
                            d = float(
                                result_obj.displacements.sel(
                                    Component=component,
                                    Node=nd,
                                )[0].values
                            )
                        all_x.append(c[0])
                        all_y.append(c[2])
                        all_v.append(d * scale * sign)
                    # Insert None to break line between elements
                    all_x.append(None)
                    all_y.append(None)
                    all_v.append(None)

                fig.add_trace(
                    go.Scatter3d(
                        x=all_x,
                        y=all_y,
                        z=all_v,
                        mode="lines",
                        line=dict(color=colour, width=3),
                        name=member,
                    )
                )
                trace_idx += 1
            except (KeyError, ValueError, IndexError):
                pass  # member not available for this model type
            continue

        for zg in range(n_groups):
            try:
                xx, zz, vals = _extract_def_data(
                    ospgrillage_obj,
                    result_obj,
                    member,
                    component,
                    loadcase,
                    z_group_num=zg,
                )
            except (KeyError, ValueError, IndexError):
                continue  # member not available for this model type
            colour = colours[trace_idx % len(colours)]
            trace_name = member if n_groups == 1 else f"{member} [{zg+1}]"
            sv = np.array(vals) * scale * sign

            fig.add_trace(
                go.Scatter3d(
                    x=xx,
                    y=zz,
                    z=sv.tolist(),
                    mode="lines+markers",
                    line=dict(color=colour, width=4),
                    marker=dict(size=3, color=colour),
                    name=trace_name,
                )
            )
            # Baseline
            fig.add_trace(
                go.Scatter3d(
                    x=xx,
                    y=zz,
                    z=[0.0] * len(vals),
                    mode="lines",
                    line=dict(color=colour, width=1, dash="dot"),
                    showlegend=False,
                )
            )
            trace_idx += 1

    # Support markers on the zero-baseline (skip for _ModelProxy)
    if show_supports and hasattr(ospgrillage_obj, "Mesh_obj"):
        supports = _extract_support_data(ospgrillage_obj)
        from collections import defaultdict

        by_type = defaultdict(list)
        for sup in supports:
            by_type[sup["fixity_type"]].append(sup)
        for ftype, sups in sorted(by_type.items()):
            symbol, colour, sz = _SUPPORT_STYLE_PLOTLY.get(ftype, ("x", "purple", 6))
            sx = [s["x"] for s in sups]
            sy = [s["z"] for s in sups]
            sz_vals = [0.0] * len(sups)  # supports sit on the baseline
            fig.add_trace(
                go.Scatter3d(
                    x=sx,
                    y=sy,
                    z=sz_vals,
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=sz,
                        color=colour,
                        line=dict(color="black", width=1),
                    ),
                    name=f"support ({ftype})",
                )
            )

    aspect = _spatial_aspect_ratio(fig)
    _no_bg = dict(showbackground=False)
    layout_kw = dict(
        scene=dict(
            xaxis=dict(title="x (m)", **_no_bg),
            yaxis=dict(title="z (m)", **_no_bg),
            zaxis=dict(title=f"displacement ({component})", **_no_bg),
            aspectmode="manual",
            aspectratio=aspect,
        ),
        legend_title="Member",
    )
    if title is _AUTO:
        layout_kw["title"] = f"Deflection ({component})"
    elif title is not None:
        layout_kw["title"] = title
    if figsize is not None:
        layout_kw["width"] = figsize[0] * 100
        layout_kw["height"] = figsize[1] * 100

    fig.update_layout(**layout_kw)
    return fig


def _plotly_3d_shell_contour(
    result_obj,
    component,
    loadcase=None,
    *,
    fig=None,
    figsize=None,
    title=_AUTO,
    colorscale="RdBu_r",
    show_colorbar=True,
    opacity=1.0,
    averaging="nodal",
):
    """Build a Plotly ``Mesh3d`` contour of a shell stress resultant.

    The returned figure can be passed as ``fig=`` to
    :func:`_plotly_3d_force` or the ``plot_bmd`` family to overlay beam
    force diagrams on top of the shell contour.

    Parameters
    ----------
    result_obj : xarray.Dataset
        Results dataset containing ``forces_shell``, ``ele_nodes_shell``,
        and ``node_coordinates``.
    component : str
        ``"Vx"``, ``"Vy"``, ``"Vz"``, ``"Mx"``, ``"My"``, or ``"Mz"``.
    loadcase : str or None
        Load case name.  ``None`` uses the first load case.
    fig : plotly.graph_objects.Figure or None
        Existing figure to add the contour to.
    figsize : tuple or None
        ``(width, height)`` in inches.
    title : str, None, or _AUTO
        Plot title.
    colorscale : str
        Any valid Plotly colorscale name.
    show_colorbar : bool
        Show colour bar legend.
    opacity : float
        Surface opacity (0–1).
    averaging : str
        Nodal averaging method (see :func:`_extract_shell_contour_data`).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _import_plotly()
    new_fig = fig is None
    if new_fig:
        fig = go.Figure()

    # Extract data — dispatch to force or displacement extraction
    if component in _DISP_COMPONENTS:
        node_values, element_quads = _extract_shell_disp_data(
            result_obj, component, loadcase,
        )
    elif component in _STRESS_RESULTANTS:
        node_values, element_quads = _extract_shell_stress_data(
            result_obj, component, loadcase,
        )
    else:
        node_values, element_quads = _extract_shell_contour_data(
            result_obj, component, loadcase, averaging=averaging,
        )

    # Build node coordinate dict from Dataset
    coords_da = result_obj["node_coordinates"]
    node_coords = {}
    for tag in coords_da.coords["Node"].values:
        node_coords[int(tag)] = coords_da.sel(Node=tag).values.tolist()

    # Triangulate
    vx, vy, vz, i_idx, j_idx, k_idx, tag_to_vidx = _triangulate_shell_mesh(
        node_coords, element_quads,
    )

    # Build intensity array aligned with vertex order
    intensity = [0.0] * len(vx)
    for tag, vidx in tag_to_vidx.items():
        intensity[vidx] = node_values.get(tag, 0.0)

    fig.add_trace(
        go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            i=i_idx,
            j=j_idx,
            k=k_idx,
            intensity=intensity,
            colorscale=colorscale,
            colorbar=dict(title=component, x=-0.05, xanchor="right") if show_colorbar else None,
            showscale=show_colorbar,
            opacity=opacity,
            flatshading=True,
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0,
            ),
            name=f"shell_{component}",
            hovertemplate=f"{component}: %{{intensity:.3g}}<extra></extra>",
        )
    )

    # Layout — only apply when creating a new figure
    if new_fig:
        _no_bg = dict(showbackground=False)
        layout_kw = dict(
            scene=dict(
                xaxis=dict(title="x (m)", **_no_bg),
                yaxis=dict(title="z (m)", **_no_bg),
                zaxis=dict(title="y (m)", **_no_bg),
                aspectmode="data",
            ),
        )
        if title is _AUTO:
            layout_kw["title"] = f"Shell Contour — {component}"
        elif title is not None:
            layout_kw["title"] = title
        if figsize is not None:
            layout_kw["width"] = figsize[0] * 100
            layout_kw["height"] = figsize[1] * 100
        fig.update_layout(**layout_kw)

    return fig


# ---------------------------------------------------------------------------
# Matplotlib plotting functions
# ---------------------------------------------------------------------------
def _plot_shell_contour_mpl(
    result_obj,
    component,
    loadcase=None,
    *,
    ax=None,
    figsize=None,
    title=_AUTO,
    cmap="RdBu_r",
    show_colorbar=True,
    averaging="nodal",
):
    """Plot a 2-D plan-view filled contour of a shell stress resultant.

    Uses :class:`matplotlib.tri.Triangulation` and
    :meth:`~matplotlib.axes.Axes.tripcolor` with Gouraud shading.

    Parameters
    ----------
    result_obj : xarray.Dataset
    component : str
    loadcase : str or None
    ax : matplotlib.axes.Axes or None
    figsize : tuple or None
    title : str, None, or _AUTO
    cmap : str
        Matplotlib colormap name.
    show_colorbar : bool
    averaging : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.tri as mtri

    if component in _DISP_COMPONENTS:
        node_values, element_quads = _extract_shell_disp_data(
            result_obj, component, loadcase,
        )
    elif component in _STRESS_RESULTANTS:
        node_values, element_quads = _extract_shell_stress_data(
            result_obj, component, loadcase,
        )
    else:
        node_values, element_quads = _extract_shell_contour_data(
            result_obj, component, loadcase, averaging=averaging,
        )

    # Build node coordinate dict
    coords_da = result_obj["node_coordinates"]
    node_coords = {}
    for tag in coords_da.coords["Node"].values:
        node_coords[int(tag)] = coords_da.sel(Node=tag).values.tolist()

    # Collect unique node tags in stable order and build index mapping
    seen = {}
    ordered_tags = []
    for quad in element_quads:
        for tag in quad:
            if tag not in seen:
                seen[tag] = len(ordered_tags)
                ordered_tags.append(tag)

    # 2D plan view: model x, model z
    x_arr = np.array([node_coords[t][0] for t in ordered_tags])
    z_arr = np.array([node_coords[t][2] for t in ordered_tags])
    vals = np.array([node_values.get(t, 0.0) for t in ordered_tags])

    # Build triangle list
    triangles = []
    for quad in element_quads:
        idxs = [seen[t] for t in quad]
        if len(idxs) == 4:
            triangles.append([idxs[0], idxs[1], idxs[2]])
            triangles.append([idxs[0], idxs[2], idxs[3]])
        elif len(idxs) == 3:
            triangles.append([idxs[0], idxs[1], idxs[2]])
    triangles = np.array(triangles, dtype=int)

    tri = mtri.Triangulation(x_arr, z_arr, triangles)

    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    tc = ax.tripcolor(tri, vals, cmap=cmap, shading="gouraud")
    if show_colorbar:
        ax.get_figure().colorbar(tc, ax=ax, label=component)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    if title is _AUTO:
        ax.set_title(f"Shell Contour — {component}")
    elif title is not None:
        ax.set_title(title)

    return ax


def plot_force(
    ospgrillage_obj,
    result_obj=None,
    component=None,
    member: str = None,
    option: str = "elements",
    loadcase: str = None,
    *,
    figsize=None,
    ax=None,
    scale: float = 1.0,
    title=_AUTO,
    color: str = "k",
    fill: bool = True,
    alpha: float = 0.4,
    show: bool = False,
    show_supports: bool = False,
):
    """
    Plot a force diagram for a grillage model result for a specified component and load case.

    .. note::
        For "shell_beam" model type, the function only plots the force diagrams for beam elements only.

    :param ospgrillage_obj: Grillage model object.
    :type ospgrillage_obj: OspGrillage
    :param result_obj: xarray DataSet of results.
    :type result_obj: xarray DataSet
    :param component: Force component to plot (e.g. ``"Mz"``, ``"Fy"``).
    :type component: str
    :param member: Member group name (required).
    :type member: str
    :param option: Element query option.
    :type option: str
    :param loadcase: Load case name.  If ``None``, uses the first load case.
    :type loadcase: str
    :param figsize: Figure size in inches ``(width, height)``.
        Ignored when *ax* is provided.
    :type figsize: tuple, optional
    :param ax: Existing matplotlib Axes to plot on.  When provided the
        function draws on this axes instead of creating a new figure.
    :type ax: :class:`~matplotlib.axes.Axes`, optional
    :param scale: Multiply plotted values by this factor (e.g. ``0.001``
        to convert N to kN).
    :type scale: float
    :param title: Plot title.  Default (``_AUTO``) uses the member name.
        Pass a string to override, or ``None`` to suppress the title.
    :type title: str or None
    :param color: Line / fill colour.
    :type color: str
    :param fill: If ``True`` (default), shade the area under the diagram.
    :type fill: bool
    :param alpha: Fill transparency (0 = transparent, 1 = opaque).
    :type alpha: float
    :param show: If ``True``, call ``plt.show()`` before returning.
    :type show: bool
    :param show_supports: If ``True``, draw support markers on the zero
        baseline.  Default ``False``.
    :type show_supports: bool
    :returns: Matplotlib axes (use ``ax.get_figure()`` to obtain the figure).
    :rtype: :class:`~matplotlib.axes.Axes`
    """
    if member is None:
        raise ValueError("Missing argument: member= is required")

    all_xx, _all_zz, all_values = _extract_force_data(
        ospgrillage_obj,
        result_obj,
        component,
        member,
        loadcase,
        option,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for xx, vals in zip(all_xx, all_values):
        scaled = vals * scale
        ax.plot(xx, scaled, "-", color=color)
        if fill:
            ax.fill_between(xx, scaled, np.zeros_like(scaled), color=color, alpha=alpha)

    if title is _AUTO:
        ax.set_title(member)
    elif title is not None:
        ax.set_title(title)
    if show_supports:
        supports = _extract_support_data(ospgrillage_obj)
        for sup in supports:
            marker, colour, sz = _SUPPORT_STYLE_MPL.get(
                sup["fixity_type"], ("D", "purple", 7)
            )
            ax.plot(
                sup["x"],
                0,
                marker=marker,
                color=colour,
                markersize=sz,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=5,
            )

    ax.set_xlabel("x (m) ")
    ax.set_ylabel(component)
    fig.tight_layout()

    if show:
        plt.show()

    return ax


def _plot_def_mpl(
    ospgrillage_obj,
    result_obj=None,
    member: str = None,
    component: str = None,
    option: str = "nodes",
    loadcase: str = None,
    *,
    figsize=None,
    ax=None,
    scale: float = 1.0,
    title=_AUTO,
    color: str = "b",
    show: bool = False,
    show_supports: bool = False,
):
    """
    Plot a displacement diagram for a grillage model result for a specified component and load case.

    .. note::
        For "shell_beam" model type, the function only plots the force diagrams for beam elements only.

    :param ospgrillage_obj: Grillage model object.
    :type ospgrillage_obj: OspGrillage
    :param result_obj: xarray DataSet of results.
    :type result_obj: xarray DataSet
    :param component: Displacement component (default ``"y"``).
    :type component: str
    :param member: Member group name (required).
    :type member: str
    :param option: Node query option, either ``"nodes"`` or ``"element"``.
    :type option: str
    :param loadcase: Load case name.  If ``None``, uses the first load case.
    :type loadcase: str
    :param figsize: Figure size in inches ``(width, height)``.
        Ignored when *ax* is provided.
    :type figsize: tuple, optional
    :param ax: Existing matplotlib Axes to plot on.  When provided the
        function draws on this axes instead of creating a new figure.
    :type ax: :class:`~matplotlib.axes.Axes`, optional
    :param scale: Multiply plotted values by this factor.
    :type scale: float
    :param title: Plot title.  Default (``_AUTO``) uses the member name.
        Pass a string to override, or ``None`` to suppress the title.
    :type title: str or None
    :param color: Line colour.
    :type color: str
    :param show: If ``True``, call ``plt.show()`` before returning.
    :type show: bool
    :returns: Matplotlib axes (use ``ax.get_figure()`` to obtain the figure).
    :rtype: :class:`~matplotlib.axes.Axes`
    """
    if member is None:
        raise ValueError("Missing argument: member= is required")

    dis_comp = component if component is not None else "y"
    xx_list, _zz_list, values_list = _extract_def_data(
        ospgrillage_obj,
        result_obj,
        member,
        dis_comp,
        loadcase,
        option,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    scaled = np.array(values_list) * scale
    ax.plot(xx_list, scaled, "-", color=color)

    if show_supports:
        supports = _extract_support_data(ospgrillage_obj)
        for sup in supports:
            marker, colour, sz = _SUPPORT_STYLE_MPL.get(
                sup["fixity_type"], ("D", "purple", 7)
            )
            ax.plot(
                sup["x"],
                0,
                marker=marker,
                color=colour,
                markersize=sz,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=5,
            )

    if title is _AUTO:
        ax.set_title(member)
    elif title is not None:
        ax.set_title(title)
    ax.set_xlabel("x (m) ")
    ax.set_ylabel(dis_comp)
    fig.tight_layout()

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# Member selection (bitflag enum)
# ---------------------------------------------------------------------------
class Members(enum.Flag):
    """Bitflag enum for selecting which grillage member groups to plot.

    Individual members can be combined with ``|``::

        og.Members.EDGE_BEAM | og.Members.INTERIOR_MAIN_BEAM

    Pre-defined composites are available for convenience::

        og.Members.LONGITUDINAL   # all four longitudinal member types
        og.Members.TRANSVERSE     # transverse slab + start/end edges
        og.Members.ALL            # everything
    """

    EDGE_BEAM = enum.auto()
    EXTERIOR_MAIN_BEAM_1 = enum.auto()
    INTERIOR_MAIN_BEAM = enum.auto()
    EXTERIOR_MAIN_BEAM_2 = enum.auto()
    START_EDGE = enum.auto()
    END_EDGE = enum.auto()
    TRANSVERSE_SLAB = enum.auto()

    # Convenience composites
    LONGITUDINAL = (
        EDGE_BEAM | EXTERIOR_MAIN_BEAM_1 | INTERIOR_MAIN_BEAM | EXTERIOR_MAIN_BEAM_2
    )
    TRANSVERSE = START_EDGE | END_EDGE | TRANSVERSE_SLAB
    ALL = LONGITUDINAL | TRANSVERSE


# Mapping between flag values and internal member name strings.
_MEMBER_NAME_MAP = {
    Members.EDGE_BEAM: "edge_beam",
    Members.EXTERIOR_MAIN_BEAM_1: "exterior_main_beam_1",
    Members.INTERIOR_MAIN_BEAM: "interior_main_beam",
    Members.EXTERIOR_MAIN_BEAM_2: "exterior_main_beam_2",
    Members.START_EDGE: "start_edge",
    Members.END_EDGE: "end_edge",
    Members.TRANSVERSE_SLAB: "transverse_slab",
}

# Reverse map: name string → flag (for backward-compatible str input).
_NAME_TO_MEMBER = {v: k for k, v in _MEMBER_NAME_MAP.items()}


def _resolve_members(member, backend="plotly"):
    """Convert a *member* argument into a list of member-name strings.

    Parameters
    ----------
    member : str, Members, or None
        * ``None`` — use default set (backend-dependent).
        * ``str``  — single member name (backward-compatible).
        * ``Members`` flag — one or more members combined with ``|``.
    backend : str
        ``"plotly"`` defaults to :attr:`Members.ALL`; ``"matplotlib"``
        defaults to :attr:`Members.LONGITUDINAL`.

    Returns
    -------
    list[str]
        Ordered list of member name strings.
    """
    if member is None:
        flag = Members.ALL if backend == "plotly" else Members.LONGITUDINAL
    elif isinstance(member, str):
        if member not in _NAME_TO_MEMBER:
            raise ValueError(
                f"Unknown member {member!r}. "
                f"Valid names: {list(_NAME_TO_MEMBER.keys())}"
            )
        flag = _NAME_TO_MEMBER[member]
    elif isinstance(member, Members):
        flag = member
    else:
        raise TypeError(
            f"member must be str, Members, or None, got {type(member).__name__}"
        )

    # Decompose the composite flag into individual members, preserving
    # the declaration order of the enum.
    return [_MEMBER_NAME_MAP[m] for m in Members if m in flag and m in _MEMBER_NAME_MAP]


# ---------------------------------------------------------------------------
# Convenience plotting wrappers
# ---------------------------------------------------------------------------


def plot_bmd(
    ospgrillage_obj,
    result_obj=None,
    members=None,
    loadcase=None,
    backend="matplotlib",
    **kwargs,
):
    """Plot bending moment diagram (Mz) for selected member groups.

    :param ospgrillage_obj: Grillage model object.
    :param result_obj: xarray DataSet of results.
    :param members: Which members to plot.

        * ``None`` (default) — all members (plotly) or longitudinal only
          (matplotlib).
        * A member name string such as ``"interior_main_beam"`` — single
          member (backward-compatible).
        * A :class:`Members` flag such as ``Members.LONGITUDINAL`` or
          ``Members.EDGE_BEAM | Members.INTERIOR_MAIN_BEAM`` — any
          combination.
    :type members: str, Members, or None
    :param loadcase: Load case name. If ``None``, uses the first load case.
    :param backend: ``"matplotlib"`` (default, static) or ``"plotly"``
        (interactive 3D).
    :param \\**kwargs: Forwarded to the underlying renderer.  See
        :func:`plot_force` (matplotlib) or the Plotly builder for accepted
        keyword arguments such as *figsize*, *ax*, *scale*, *title*,
        *color*, *fill*, *alpha*, and *show*.
    :returns: Single axes when *members* is a string, else list of axes.
        For ``backend="plotly"``, returns a single
        :class:`plotly.graph_objects.Figure`.
    """
    resolved = _resolve_members(members, backend)
    if backend == "plotly":
        show = kwargs.pop("show", True)
        plotly_kw = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "figsize",
                "scale",
                "title",
                "alpha",
                "fill",
                "fill_alpha",
                "show_supports",
            )
        }
        plotly_kw["fig"] = kwargs.get("ax", None)
        fig = _plotly_3d_force(
            ospgrillage_obj,
            result_obj,
            component="Mz",
            members=resolved,
            loadcase=loadcase,
            comp_label="Mz",
            **plotly_kw,
        )
        if show:
            _show_plotly_fig(fig)
            return None
        return fig
    # matplotlib path — single string returns one axes (backward compat)
    if isinstance(members, str):
        return plot_force(
            ospgrillage_obj,
            result_obj,
            component="Mz",
            member=members,
            loadcase=loadcase,
            **kwargs,
        )
    figs = []
    for m in resolved:
        try:
            figs.append(
                plot_force(
                    ospgrillage_obj,
                    result_obj,
                    component="Mz",
                    member=m,
                    loadcase=loadcase,
                    **kwargs,
                )
            )
        except (ValueError, KeyError, IndexError):
            pass
    return figs


def plot_sfd(
    ospgrillage_obj,
    result_obj=None,
    members=None,
    loadcase=None,
    backend="matplotlib",
    **kwargs,
):
    """Plot shear force diagram (Fy) for selected member groups.

    :param ospgrillage_obj: Grillage model object.
    :param result_obj: xarray DataSet of results.
    :param members: Which members to plot.

        * ``None`` (default) — all members (plotly) or longitudinal only
          (matplotlib).
        * A member name string such as ``"interior_main_beam"`` — single
          member (backward-compatible).
        * A :class:`Members` flag such as ``Members.LONGITUDINAL`` or
          ``Members.EDGE_BEAM | Members.INTERIOR_MAIN_BEAM`` — any
          combination.
    :type members: str, Members, or None
    :param loadcase: Load case name. If ``None``, uses the first load case.
    :param backend: ``"matplotlib"`` (default, static) or ``"plotly"``
        (interactive 3D).
    :param \\**kwargs: Forwarded to the underlying renderer.  See
        :func:`plot_force` for accepted keyword arguments.
    :returns: Single axes when *members* is a string, else list of axes.
        For ``backend="plotly"``, returns a single
        :class:`plotly.graph_objects.Figure`.
    """
    resolved = _resolve_members(members, backend)
    if backend == "plotly":
        show = kwargs.pop("show", True)
        plotly_kw = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "figsize",
                "scale",
                "title",
                "alpha",
                "fill",
                "fill_alpha",
                "show_supports",
            )
        }
        plotly_kw["fig"] = kwargs.get("ax", None)
        fig = _plotly_3d_force(
            ospgrillage_obj,
            result_obj,
            component="Fy",
            members=resolved,
            loadcase=loadcase,
            comp_label="Fy",
            **plotly_kw,
        )
        if show:
            _show_plotly_fig(fig)
            return None
        return fig
    # matplotlib path — single string returns one axes (backward compat)
    if isinstance(members, str):
        return plot_force(
            ospgrillage_obj,
            result_obj,
            component="Fy",
            member=members,
            loadcase=loadcase,
            **kwargs,
        )
    figs = []
    for m in resolved:
        try:
            figs.append(
                plot_force(
                    ospgrillage_obj,
                    result_obj,
                    component="Fy",
                    member=m,
                    loadcase=loadcase,
                    **kwargs,
                )
            )
        except (ValueError, KeyError, IndexError):
            pass
    return figs


def plot_def(
    ospgrillage_obj,
    result_obj=None,
    members=None,
    loadcase=None,
    backend="matplotlib",
    **kwargs,
):
    """Plot vertical deflection (y-displacement) for selected member groups.

    :param ospgrillage_obj: Grillage model object.
    :param result_obj: xarray DataSet of results.
    :param members: Which members to plot.

        * ``None`` (default) — all members (plotly) or longitudinal only
          (matplotlib).
        * A member name string such as ``"interior_main_beam"`` — single
          member (backward-compatible).
        * A :class:`Members` flag such as ``Members.LONGITUDINAL`` or
          ``Members.EDGE_BEAM | Members.INTERIOR_MAIN_BEAM`` — any
          combination.
    :type members: str, Members, or None
    :param loadcase: Load case name. If ``None``, uses the first load case.
    :param backend: ``"matplotlib"`` (default, static) or ``"plotly"``
        (interactive 3D).
    :param \\**kwargs: Forwarded to the underlying renderer.  See
        :func:`plot_force` for accepted keyword arguments.
    :returns: Single axes when *members* is a string, else list of axes.
        For ``backend="plotly"``, returns a single
        :class:`plotly.graph_objects.Figure`.
    """
    resolved = _resolve_members(members, backend)
    if backend == "plotly":
        show = kwargs.pop("show", True)
        plotly_kw = {
            k: v
            for k, v in kwargs.items()
            if k in ("figsize", "scale", "title", "show_supports")
        }
        plotly_kw["fig"] = kwargs.get("ax", None)
        fig = _plotly_3d_def(
            ospgrillage_obj,
            result_obj,
            members=resolved,
            component="y",
            loadcase=loadcase,
            **plotly_kw,
        )
        if show:
            _show_plotly_fig(fig)
            return None
        return fig
    # matplotlib path — filter to def-compatible kwargs (no fill/alpha)
    def_kw = {
        k: v
        for k, v in kwargs.items()
        if k in ("figsize", "ax", "scale", "title", "color", "show", "show_supports")
    }
    # Single string returns one axes (backward compat)
    if isinstance(members, str):
        return _plot_def_mpl(
            ospgrillage_obj,
            result_obj,
            member=members,
            component="y",
            loadcase=loadcase,
            **def_kw,
        )
    figs = []
    for m in resolved:
        try:
            figs.append(
                _plot_def_mpl(
                    ospgrillage_obj,
                    result_obj,
                    member=m,
                    component="y",
                    loadcase=loadcase,
                    **def_kw,
                )
            )
        except (ValueError, KeyError, IndexError):
            pass
    return figs


def plot_tmd(
    ospgrillage_obj,
    result_obj=None,
    members=None,
    loadcase=None,
    backend="matplotlib",
    **kwargs,
):
    """Plot torsion moment diagram (Mx) for selected member groups.

    :param ospgrillage_obj: Grillage model object.
    :param result_obj: xarray DataSet of results.
    :param members: Which members to plot.

        * ``None`` (default) — all members (plotly) or longitudinal only
          (matplotlib).
        * A member name string such as ``"interior_main_beam"`` — single
          member (backward-compatible).
        * A :class:`Members` flag such as ``Members.LONGITUDINAL`` or
          ``Members.EDGE_BEAM | Members.INTERIOR_MAIN_BEAM`` — any
          combination.
    :type members: str, Members, or None
    :param loadcase: Load case name. If ``None``, uses the first load case.
    :param backend: ``"matplotlib"`` (default, static) or ``"plotly"``
        (interactive 3D).
    :param \\**kwargs: Forwarded to the underlying renderer.  See
        :func:`plot_force` (matplotlib) or the Plotly builder for accepted
        keyword arguments such as *figsize*, *ax*, *scale*, *title*,
        *color*, *fill*, *alpha*, and *show*.
    :returns: Single axes when *members* is a string, else list of axes.
        For ``backend="plotly"``, returns a single
        :class:`plotly.graph_objects.Figure`.
    """
    resolved = _resolve_members(members, backend)
    if backend == "plotly":
        show = kwargs.pop("show", True)
        plotly_kw = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "figsize",
                "scale",
                "title",
                "alpha",
                "fill",
                "fill_alpha",
                "show_supports",
            )
        }
        plotly_kw["fig"] = kwargs.get("ax", None)
        fig = _plotly_3d_force(
            ospgrillage_obj,
            result_obj,
            component="Mx",
            members=resolved,
            loadcase=loadcase,
            comp_label="Mx",
            **plotly_kw,
        )
        if show:
            _show_plotly_fig(fig)
            return None
        return fig
    # matplotlib path — single string returns one axes (backward compat)
    if isinstance(members, str):
        return plot_force(
            ospgrillage_obj,
            result_obj,
            component="Mx",
            member=members,
            loadcase=loadcase,
            **kwargs,
        )
    figs = []
    for m in resolved:
        try:
            figs.append(
                plot_force(
                    ospgrillage_obj,
                    result_obj,
                    component="Mx",
                    member=m,
                    loadcase=loadcase,
                    **kwargs,
                )
            )
        except (ValueError, KeyError, IndexError):
            pass
    return figs


# ---------------------------------------------------------------------------
# Shell contour convenience wrapper
# ---------------------------------------------------------------------------
def plot_srf(
    result_obj,
    component="Mx",
    loadcase=None,
    backend="plotly",
    **kwargs,
):
    """Plot a contour field over the shell mesh.

    Visualises shell element data as an interpolated contour over the
    deck slab mesh of a ``shell_beam`` model.  Three families of
    component are supported:

    * **Shell forces** — ``Vx``, ``Vy``, ``Vz``, ``Mx``, ``My``,
      ``Mz``.  Element end forces averaged at shared nodes.
    * **Displacements** — ``Dx``, ``Dy``, ``Dz``.  Nodal translations
      read directly from the ``displacements`` array.
    * **Stress resultants** — ``N11``, ``N22``, ``N12``, ``M11``,
      ``M22``, ``M12``, ``Q13``, ``Q23``.  Section stress resultants
      averaged across the 4 Gauss points per element, then averaged
      at shared nodes.

    The returned Plotly figure can be composed with beam force
    diagrams by passing it as ``ax=`` to :func:`plot_bmd` etc.::

        fig = og.plot_srf(results, "Mx", backend="plotly", show=False)
        og.plot_bmd(proxy, results, backend="plotly", ax=fig)

    Parameters
    ----------
    result_obj : xarray.Dataset
        Results dataset.  Must contain ``ele_nodes_shell`` and
        ``node_coordinates``.  Depending on *component*, also requires
        ``forces_shell`` (shell forces), ``displacements``
        (displacement components), or ``stresses_shell`` (stress
        resultants).
    component : str, default ``"Mx"``
        Component to plot.  One of:

        * Shell forces: ``"Vx"``, ``"Vy"``, ``"Vz"``, ``"Mx"``,
          ``"My"``, ``"Mz"``
        * Displacements: ``"Dx"``, ``"Dy"``, ``"Dz"``
        * Stress resultants: ``"N11"``, ``"N22"``, ``"N12"``,
          ``"M11"``, ``"M22"``, ``"M12"``, ``"Q13"``, ``"Q23"``
    loadcase : str or None, default ``None``
        Load case name to plot.  ``None`` uses the first load case.
    backend : str, default ``"plotly"``
        ``"plotly"`` for interactive 3-D or ``"matplotlib"`` for 2-D.
    colorscale : str, default ``"RdBu_r"``
        Plotly colorscale name (or matplotlib *cmap* name).  Use a
        diverging scale (e.g. ``"RdBu_r"``) for signed data and a
        sequential scale (e.g. ``"Viridis"``) for displacements.
    show : bool
        Display the plot immediately.  Default ``True`` for plotly,
        ``False`` for matplotlib.
    kwargs : dict, optional
        Forwarded to the backend renderer.  Accepted keys include
        *figsize*, *title*, *opacity*, *show_colorbar*, *averaging*,
        and *ax* / *fig* for composing onto an existing figure.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.axes.Axes
        The figure object, or ``None`` when *show* is ``True`` (plotly
        backend only).

    Raises
    ------
    ValueError
        If the dataset does not contain the required data variables
        for the requested *component*.
    """
    if component not in _SRF_COMPONENTS:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Expected one of {_SRF_COMPONENTS}."
        )
    if "ele_nodes_shell" not in result_obj:
        raise ValueError(
            "Dataset does not contain shell element results. "
            "plot_srf() requires a shell_beam model."
        )
    if component in _DISP_COMPONENTS and "displacements" not in result_obj:
        raise ValueError("Dataset does not contain 'displacements'.")
    if component in _SHELL_COMPONENTS and "forces_shell" not in result_obj:
        raise ValueError("Dataset does not contain 'forces_shell'.")
    if component in _STRESS_RESULTANTS and "stresses_shell" not in result_obj:
        raise ValueError(
            "Dataset does not contain 'stresses_shell'. "
            "Re-run analysis with the latest ospgrillage to include "
            "shell section stress resultants."
        )
    if "node_coordinates" not in result_obj:
        raise ValueError(
            "Dataset does not contain 'node_coordinates'. "
            "Re-generate results with the latest ospgrillage."
        )

    if backend == "plotly":
        show = kwargs.pop("show", True)
        plotly_kw = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "figsize",
                "title",
                "colorscale",
                "show_colorbar",
                "opacity",
                "averaging",
            )
        }
        # Accept 'ax' as alias for 'fig' (matches plot_bmd pattern)
        plotly_kw["fig"] = kwargs.get("ax", kwargs.get("fig", None))
        fig = _plotly_3d_shell_contour(
            result_obj,
            component,
            loadcase,
            **plotly_kw,
        )
        if show:
            _show_plotly_fig(fig)
            return None
        return fig

    # matplotlib path
    mpl_kw = {
        k: v
        for k, v in kwargs.items()
        if k in ("ax", "figsize", "title", "show_colorbar", "averaging")
    }
    # Map 'colorscale' -> 'cmap' for matplotlib
    mpl_kw["cmap"] = kwargs.get("colorscale", kwargs.get("cmap", "RdBu_r"))
    show = kwargs.get("show", False)
    ax = _plot_shell_contour_mpl(
        result_obj,
        component,
        loadcase,
        **mpl_kw,
    )
    if show:
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Model geometry plotting
# ---------------------------------------------------------------------------

# Support type classification from fixity vectors.
_FIXITY_CLASSIFICATION = {
    (1, 1, 1, 0, 0, 0): "pin",
    (0, 1, 1, 0, 0, 0): "roller",
    (1, 1, 1, 1, 1, 1): "fixed",
}

# Visual style per support type — (marker, colour, size).
_SUPPORT_STYLE_MPL = {
    "pin": ("^", "green", 10),
    "roller": ("o", "orange", 8),
    "fixed": ("s", "red", 8),
    "spring": ("v", "magenta", 9),
    "other": ("D", "purple", 7),
}
_SUPPORT_STYLE_PLOTLY = {
    "pin": ("diamond", "green", 8),
    "roller": ("circle", "orange", 7),
    "fixed": ("square", "red", 8),
    "spring": ("diamond-open", "magenta", 8),
    "other": ("x", "purple", 6),
}


def _extract_support_data(grillage_obj):
    """Return node positions and support types for boundary-condition drawing.

    Mirrors the logic of
    :meth:`~ospgrillage.osp_grillage.OspGrillage._write_op_fix` so the
    visualisation matches the actual constraints applied to the model.

    Returns
    -------
    list[dict]
        Each dict has keys ``node_tag``, ``x``, ``y``, ``z``,
        ``fixity_type`` (``"pin"`` / ``"roller"`` / ``"fixed"`` /
        ``"other"``).
    """
    if not hasattr(grillage_obj, "Mesh_obj"):
        return []
    mesh = grillage_obj.Mesh_obj
    node_spec = mesh.node_spec
    support_type_dict = getattr(grillage_obj, "edge_support_type_dict", {})

    # Determine which attribute stores the node → edge-group mapping.
    edge_recorder = getattr(mesh, "edge_node_recorder", {})

    # Nodes to skip — mirrors the condition in _write_op_fix:
    #   z_group in common_z_group_element[0]  →  pass (no fix applied)
    skip_z_groups = set()
    czge = getattr(mesh, "common_z_group_element", None)
    if czge and 0 in czge:
        skip_z_groups = set(czge[0])

    # Identify nodes that have spring supports.  spring_node_pairs maps
    # support_node → original_node; the *support* nodes are the ones that
    # carry the fixity and sit in edge_node_recorder after set_spring_support.
    spring_support_nodes = set(getattr(grillage_obj, "spring_node_pairs", {}).keys())

    supports = []
    for node_tag, edge_group_num in edge_recorder.items():
        if node_tag not in node_spec:
            continue
        # Skip nodes that _write_op_fix skips.
        z_grp = node_spec[node_tag].get("z_group")
        if z_grp is not None and z_grp in skip_z_groups:
            continue

        coord = node_spec[node_tag]["coordinate"]

        if node_tag in spring_support_nodes:
            fixity_type = "spring"
        else:
            fixity_vec = support_type_dict.get(edge_group_num, [0] * 6)
            fixity_tuple = tuple(int(v) for v in fixity_vec)
            fixity_type = _FIXITY_CLASSIFICATION.get(fixity_tuple, "other")

        supports.append(
            {
                "node_tag": node_tag,
                "x": coord[0],
                "y": coord[1],
                "z": coord[2],
                "fixity_type": fixity_type,
            }
        )

    return supports


# Colour palette for member groups (one per member type).
_MEMBER_COLOURS = {
    "edge_beam": "grey",
    "exterior_main_beam_1": "blue",
    "interior_main_beam": "green",
    "exterior_main_beam_2": "blue",
    "start_edge": "red",
    "end_edge": "red",
    "transverse_slab": "orange",
}
_DEFAULT_COLOUR = "black"


def _extract_mesh_data(grillage_obj):
    """Return element geometry grouped by member name.

    Returns
    -------
    data : dict[str, list[tuple]]
        ``{member_name: [(node_i_xyz, node_j_xyz, ele_tag, node_i_tag, node_j_tag), ...]}``
    all_nodes : dict[int, list]
        ``{node_tag: [x, y, z]}``
    quads : list[tuple]
        ``[(c1, c2, c3, c4), ...]`` where each ``ci`` is ``[x, y, z]``.
        Non-empty only for shell-type meshes.
    """
    data = {}  # member_name -> list of (coord_i, coord_j, ele_tag, node_i_tag, node_j_tag)
    all_nodes = {}  # node_tag -> [x, y, z]
    if hasattr(grillage_obj, "Mesh_obj"):
        mesh = grillage_obj.Mesh_obj
        node_spec = mesh.node_spec
        z_group_map = grillage_obj.common_grillage_element_z_group

        def _add_elements(member_name, ele_list):
            entries = data.setdefault(member_name, [])
            for ele in ele_list:
                tag, ni, nj = ele[0], ele[1], ele[2]
                ci = node_spec[ni]["coordinate"]
                cj = node_spec[nj]["coordinate"]
                entries.append((ci, cj, tag, int(ni), int(nj)))
                all_nodes[ni] = ci
                all_nodes[nj] = cj

        for member_name in [
            "edge_beam",
            "exterior_main_beam_1",
            "interior_main_beam",
            "exterior_main_beam_2",
        ]:
            if member_name not in z_group_map:
                continue
            for z_idx in z_group_map[member_name]:
                if z_idx in mesh.z_group_to_ele:
                    _add_elements(member_name, mesh.z_group_to_ele[z_idx])

        for member_name in ["start_edge", "end_edge"]:
            if member_name not in z_group_map:
                continue
            for edge_idx in z_group_map[member_name]:
                if edge_idx in mesh.edge_group_to_ele:
                    _add_elements(member_name, mesh.edge_group_to_ele[edge_idx])

        _add_elements("transverse_slab", mesh.trans_ele)

        quads = []
        links = []
        if hasattr(grillage_obj, "shell_element_command_list"):
            grid = getattr(mesh, "grid_number_dict", {})
            for node_list in grid.values():
                valid = [n for n in node_list if isinstance(n, (int, float))]
                if len(valid) >= 3:
                    coords = [node_spec[n]["coordinate"] for n in valid]
                    quads.append(tuple(coords))
                    for n in valid:
                        all_nodes[n] = node_spec[n]["coordinate"]

            link_dict = getattr(mesh, "link_dict", {})
            for beam_node, slab_nodes in link_dict.items():
                bc = node_spec[beam_node]["coordinate"]
                all_nodes[beam_node] = bc
                for sn in slab_nodes:
                    sc = node_spec[sn]["coordinate"]
                    links.append((sc, bc))
                    all_nodes[sn] = sc

        return data, all_nodes, quads, links

    if isinstance(grillage_obj, _ModelProxy):
        node_spec = grillage_obj._node_spec
        element_node_map = getattr(grillage_obj, "_element_nodes", {})

        def _flatten_numeric(values):
            flat = []
            stack = list(values) if isinstance(values, list) else [values]
            while stack:
                item = stack.pop(0)
                if isinstance(item, list):
                    stack = list(item) + stack
                    continue
                try:
                    flat.append(int(item))
                except (TypeError, ValueError):
                    continue
            return flat

        for member_name, info in grillage_obj._members.items():
            entries = data.setdefault(member_name, [])
            element_groups = info.get("elements", [])
            node_groups = info.get("nodes", [])
            for group_idx, nodes in enumerate(node_groups):
                elements = (
                    element_groups[group_idx] if group_idx < len(element_groups) else []
                )
                element_ids = _flatten_numeric(elements)

                added_from_elements = False
                if element_ids and element_node_map:
                    for etag in element_ids:
                        node_pair = element_node_map.get(int(etag))
                        if not node_pair or len(node_pair) < 2:
                            continue
                        ni, nj = int(node_pair[0]), int(node_pair[1])
                        if ni not in node_spec or nj not in node_spec:
                            continue
                        ci = node_spec[ni]["coordinate"]
                        cj = node_spec[nj]["coordinate"]
                        entries.append((ci, cj, int(etag), ni, nj))
                        all_nodes[ni] = ci
                        all_nodes[nj] = cj
                        added_from_elements = True
                if added_from_elements:
                    continue

                if nodes and isinstance(nodes, list) and len(nodes) == 1 and isinstance(nodes[0], list):
                    nodes = nodes[0]
                if not nodes or len(nodes) < 2:
                    continue

                if (
                    isinstance(nodes[0], list)
                    and all(isinstance(seg, list) and len(seg) >= 2 for seg in nodes)
                ):
                    for seg_idx, seg in enumerate(nodes):
                        ni = int(seg[0])
                        nj = int(seg[1])
                        if ni not in node_spec or nj not in node_spec:
                            continue
                        ci = node_spec[ni]["coordinate"]
                        cj = node_spec[nj]["coordinate"]
                        tag = int(elements[seg_idx]) if seg_idx < len(elements) else seg_idx + 1
                        entries.append((ci, cj, tag, ni, nj))
                        all_nodes[ni] = ci
                        all_nodes[nj] = cj
                    continue

                for seg_idx, (ni, nj) in enumerate(zip(nodes[:-1], nodes[1:])):
                    ni = int(ni)
                    nj = int(nj)
                    if ni not in node_spec or nj not in node_spec:
                        continue
                    ci = node_spec[ni]["coordinate"]
                    cj = node_spec[nj]["coordinate"]
                    tag = int(elements[seg_idx]) if seg_idx < len(elements) else seg_idx + 1
                    entries.append((ci, cj, tag, ni, nj))
                    all_nodes[ni] = ci
                    all_nodes[nj] = cj
        quads = []
        for shell_nodes in getattr(grillage_obj, "_shell_element_nodes", []):
            coords = []
            valid = True
            for node_tag in shell_nodes:
                node_info = node_spec.get(int(node_tag))
                if node_info is None:
                    valid = False
                    break
                coord = node_info["coordinate"]
                coords.append(coord)
                all_nodes[int(node_tag)] = coord
            if valid and len(coords) >= 3:
                quads.append(tuple(coords))
        return data, all_nodes, quads, []

    raise TypeError("Unsupported grillage object for mesh extraction")


def _filter_member_geometry(data, members):
    """Filter extracted mesh geometry to selected member groups."""
    if members is None:
        return data
    selected_names = set(_resolve_members(members, backend="plotly"))
    return {name: elements for name, elements in data.items() if name in selected_names}


def _visible_node_ids_from_member_data(data):
    """Return node tags referenced by currently visible member line segments."""
    node_ids = set()
    for elements in data.values():
        for element_entry in elements:
            if len(element_entry) >= 5:
                node_ids.add(int(element_entry[3]))
                node_ids.add(int(element_entry[4]))
    return node_ids


def _plot_model_matplotlib(
    grillage_obj,
    *,
    figsize=None,
    ax=None,
    members=None,
    title=_AUTO,
    show_nodes=True,
    show_node_labels=False,
    show_element_labels=False,
    color_by_member=True,
    show_supports=True,
    show_rigid_links=True,
    show=False,
):
    """Render a 2-D plan view (x vs z) of the grillage mesh."""
    data, all_nodes, quads, links = _extract_mesh_data(grillage_obj)
    data = _filter_member_geometry(data, members)
    if members is not None:
        visible_nodes = _visible_node_ids_from_member_data(data)
        all_nodes = {
            node_tag: coord
            for node_tag, coord in all_nodes.items()
            if node_tag in visible_nodes
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw shell quad patches (if shell model)
    if quads:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        patches = []
        for coords in quads:
            # coords is tuple of [x,y,z] lists — plot in x-z plane
            verts = [(c[0], c[2]) for c in coords]
            patches.append(Polygon(verts, closed=True))
        pc = PatchCollection(
            patches,
            facecolor="lightblue",
            edgecolor="steelblue",
            linewidth=0.4,
            alpha=0.5,
        )
        ax.add_collection(pc)

    # Draw beam elements grouped by member
    for member_name, elements in data.items():
        colour = (
            _MEMBER_COLOURS.get(member_name, _DEFAULT_COLOUR)
            if color_by_member
            else "k"
        )
        for element_entry in elements:
            ci, cj, etag = element_entry[:3]
            ax.plot([ci[0], cj[0]], [ci[2], cj[2]], "-", color=colour, linewidth=1)
            if show_element_labels:
                mx = 0.5 * (ci[0] + cj[0])
                mz = 0.5 * (ci[2] + cj[2])
                ax.text(
                    mx, mz, str(etag), fontsize=6, color="red", ha="center", va="center"
                )

    # Draw nodes
    if show_nodes or show_node_labels:
        for ntag, coord in all_nodes.items():
            if show_nodes:
                ax.plot(coord[0], coord[2], ".", color="k", markersize=3)
            if show_node_labels:
                ax.text(
                    coord[0],
                    coord[2],
                    f" {ntag}",
                    fontsize=5,
                    color="blue",
                    ha="left",
                    va="bottom",
                )

    # Draw support symbols
    if show_supports:
        supports = _extract_support_data(grillage_obj)
        plotted_types = set()
        for sup in supports:
            marker, colour, sz = _SUPPORT_STYLE_MPL.get(
                sup["fixity_type"], ("D", "purple", 7)
            )
            ax.plot(
                sup["x"],
                sup["z"],
                marker=marker,
                color=colour,
                markersize=sz,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="none",
                zorder=5,
            )
            plotted_types.add(sup["fixity_type"])

    # Add legends — members and supports are separated for clarity.
    from matplotlib.lines import Line2D

    member_handles = []
    if quads:
        member_handles.append(
            Line2D(
                [0],
                [0],
                color="steelblue",
                linewidth=1,
                marker="s",
                markerfacecolor="lightblue",
                markersize=6,
                label="shell_slab",
            )
        )
    for member_name in data:
        colour = (
            _MEMBER_COLOURS.get(member_name, _DEFAULT_COLOUR)
            if color_by_member
            else "k"
        )
        member_handles.append(
            Line2D([0], [0], color=colour, linewidth=1, label=member_name)
        )
    if member_handles and color_by_member:
        leg1 = ax.legend(
            handles=member_handles,
            fontsize=7,
            loc="upper left",
            title="Members",
            title_fontsize=7,
        )
        ax.add_artist(leg1)

    if show_supports and plotted_types:
        sup_handles = []
        for ftype in sorted(plotted_types):
            marker, colour, sz = _SUPPORT_STYLE_MPL.get(ftype, ("D", "purple", 7))
            sup_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=marker,
                    color=colour,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=sz,
                    label=ftype,
                )
            )
        ax.legend(
            handles=sup_handles,
            fontsize=7,
            loc="upper right",
            title="Supports",
            title_fontsize=7,
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")

    if title is _AUTO:
        ax.set_title("Grillage Model")
    elif title is not None:
        ax.set_title(title)

    fig.tight_layout()
    if show:
        plt.show()

    return ax


def _plot_model_plotly(
    grillage_obj,
    *,
    fig=None,
    figsize=None,
    members=None,
    title=_AUTO,
    show_nodes=True,
    show_node_labels=False,
    show_element_labels=False,
    color_by_member=True,
    show_supports=True,
    show_rigid_links=True,
    show=False,
):
    """Render an interactive 3-D model view with Plotly."""
    go = _import_plotly()
    if fig is None:
        fig = go.Figure()

    data, all_nodes, quads, links = _extract_mesh_data(grillage_obj)
    data = _filter_member_geometry(data, members)
    if members is not None:
        visible_nodes = _visible_node_ids_from_member_data(data)
        all_nodes = {
            node_tag: coord
            for node_tag, coord in all_nodes.items()
            if node_tag in visible_nodes
        }

    colours = ["grey", "blue", "green", "blue", "red", "red", "orange"]
    colour_map = dict(zip(_MEMBER_COLOURS.keys(), colours))

    # Draw elements grouped by member
    for member_name, elements in data.items():
        colour = colour_map.get(member_name, "black") if color_by_member else "black"
        xs, ys, zs, hover_text = [], [], [], []
        for element_entry in elements:
            ci, cj, etag = element_entry[:3]
            ni = int(element_entry[3]) if len(element_entry) > 3 else None
            nj = int(element_entry[4]) if len(element_entry) > 4 else None
            xs.extend([ci[0], cj[0], None])
            ys.extend([ci[2], cj[2], None])
            zs.extend([-ci[1], -cj[1], None])
            if ni is not None and nj is not None:
                text = f"member: {member_name}<br>element: {etag}<br>nodes: {ni}-{nj}"
            else:
                text = f"member: {member_name}<br>element: {etag}"
            hover_text.extend([text, text, None])
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=colour, width=3),
                name=member_name,
                text=hover_text,
                hovertemplate=(
                    "%{text}<br>"
                    "x: %{x:.3f}<br>"
                    "z: %{y:.3f}<br>"
                    "plot y: %{z:.3f}<extra></extra>"
                ),
            )
        )

    # Shell quad surface (if shell model)
    if quads:
        # Build a flat vertex list and triangle indices (each quad → 2 triangles)
        vx, vy, vz = [], [], []
        i_idx, j_idx, k_idx = [], [], []
        base = 0
        for coords in quads:
            for c in coords:
                vx.append(c[0])
                vy.append(c[2])  # z → plotly y
                vz.append(-c[1])  # y → plotly z (negated: beams below slab)
            n = len(coords)
            if n == 4:
                i_idx.extend([base, base])
                j_idx.extend([base + 1, base + 2])
                k_idx.extend([base + 2, base + 3])
            elif n == 3:
                i_idx.append(base)
                j_idx.append(base + 1)
                k_idx.append(base + 2)
            base += n
        fig.add_trace(
            go.Mesh3d(
                x=vx,
                y=vy,
                z=vz,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color="lightblue",
                opacity=0.5,
                name="shell_slab",
            )
        )

    # Rigid links (slab ↔ offset beam)
    if show_rigid_links and links:
        lx, ly, lz = [], [], []
        for sc, bc in links:
            lx.extend([sc[0], bc[0], None])
            ly.extend([sc[2], bc[2], None])  # z → plotly y
            lz.extend([-sc[1], -bc[1], None])  # y → plotly z (negated)
        fig.add_trace(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="lines",
                line=dict(color="grey", width=1),
                name="rigid_link",
                showlegend=True,
            )
        )

    # Node markers
    if show_nodes:
        ntags = list(all_nodes.keys())
        nx = [all_nodes[n][0] for n in ntags]
        ny = [all_nodes[n][2] for n in ntags]
        nz = [-all_nodes[n][1] for n in ntags]
        text = [str(n) for n in ntags]
        node_y = [all_nodes[n][1] for n in ntags]
        mode = "markers+text" if show_node_labels else "markers"
        fig.add_trace(
            go.Scatter3d(
                x=nx,
                y=ny,
                z=nz,
                mode=mode,
                marker=dict(size=2, color="black"),
                text=text,
                customdata=np.asarray(node_y, dtype=float),
                textposition="top center",
                textfont=dict(size=8),
                name="nodes",
                showlegend=False,
                hovertemplate=(
                    "node: %{text}<br>"
                    "x: %{x:.3f}<br>"
                    "z: %{y:.3f}<br>"
                    "y: %{customdata:.3f}<extra></extra>"
                ),
            )
        )

    # Element labels at midpoints
    if show_element_labels:
        ex, ey, ez, etexts = [], [], [], []
        for member_name, elements in data.items():
            for element_entry in elements:
                ci, cj, etag = element_entry[:3]
                ex.append(0.5 * (ci[0] + cj[0]))
                ey.append(0.5 * (ci[2] + cj[2]))
                ez.append(-0.5 * (ci[1] + cj[1]))
                etexts.append(str(etag))
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="text",
                text=etexts,
                textfont=dict(size=7, color="red"),
                showlegend=False,
            )
        )

    # Support markers
    if show_supports:
        supports = _extract_support_data(grillage_obj)
        from collections import defaultdict

        by_type = defaultdict(list)
        for sup in supports:
            by_type[sup["fixity_type"]].append(sup)
        for ftype, sups in sorted(by_type.items()):
            symbol, colour, sz = _SUPPORT_STYLE_PLOTLY.get(ftype, ("x", "purple", 6))
            sx = [s["x"] for s in sups]
            sy = [s["z"] for s in sups]  # model z → plotly y
            sz_vals = [-s["y"] for s in sups]  # model y → plotly z (negated)
            fig.add_trace(
                go.Scatter3d(
                    x=sx,
                    y=sy,
                    z=sz_vals,
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=sz,
                        color=colour,
                        line=dict(color="black", width=1),
                    ),
                    name=f"support ({ftype})",
                )
            )

    # Layout — equal axis scaling (true proportions)
    _no_bg = dict(showbackground=False)
    layout_kw = dict(
        scene=dict(
            xaxis=dict(title="x (m)", **_no_bg),
            yaxis=dict(title="z (m)", **_no_bg),
            zaxis=dict(title="y (m)", **_no_bg),
            aspectmode="data",
        ),
        legend_title="Member",
    )
    if title is _AUTO:
        layout_kw["title"] = "Grillage Model"
    elif title is not None:
        layout_kw["title"] = title
    if figsize is not None:
        layout_kw["width"] = figsize[0] * 100
        layout_kw["height"] = figsize[1] * 100

    fig.update_layout(**layout_kw)

    if show:
        _show_plotly_fig(fig)
        return None  # avoid Jupyter double-display via _repr_html_

    return fig


def plot_model(grillage_obj, *, backend="matplotlib", **kwargs):
    """Plot the grillage mesh geometry.

    :param grillage_obj: Grillage model (must have been created with
        :meth:`~ospgrillage.osp_grillage.OspGrillage.create_osp_model`).
    :type grillage_obj: OspGrillage
    :param backend: ``"matplotlib"`` (default, 2-D plan view) or
        ``"plotly"`` (interactive 3-D).
    :type backend: str
    :param figsize: Figure size in inches ``(width, height)``.
        Ignored when *ax* is provided.
    :type figsize: tuple, optional
    :param ax: Existing matplotlib Axes to plot on (matplotlib backend only).
    :type ax: :class:`~matplotlib.axes.Axes`, optional
    :param fig: Existing Plotly Figure to add traces to (plotly backend only).
    :type fig: :class:`plotly.graph_objects.Figure`, optional
    :param title: Plot title.  Default auto-generates ``"Grillage Model"``.
        Pass a string to override, or ``None`` to suppress.
    :type title: str or None
    :param show_nodes: Show node markers.  Default ``True``.
    :type show_nodes: bool
    :param show_node_labels: Annotate node tags.  Default ``False``.
    :type show_node_labels: bool
    :param show_element_labels: Annotate element tags.  Default ``False``.
    :type show_element_labels: bool
    :param color_by_member: Colour elements by member group.  Default ``True``.
    :type color_by_member: bool
    :param show_supports: Draw support symbols at restrained nodes.
        Default ``True``.
    :type show_supports: bool
    :param show_rigid_links: Draw rigid-link connections (shell-beam models
        only).  Default ``True``.
    :type show_rigid_links: bool
    :param show: If ``True``, display the plot immediately.
        Defaults to ``True`` for plotly, ``False`` for matplotlib
        (matplotlib inline backends auto-display).
    :type show: bool
    :returns: Matplotlib axes (use ``ax.get_figure()`` for the figure) or
        Plotly figure (``None`` when ``show=True``).
    :rtype: :class:`~matplotlib.axes.Axes` or
        :class:`plotly.graph_objects.Figure` or None
    """
    if backend == "plotly":
        plotly_kw = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "fig",
                "figsize",
                "members",
                "title",
                "show_nodes",
                "show_node_labels",
                "show_element_labels",
                "color_by_member",
                "show_supports",
                "show_rigid_links",
                "show",
            )
        }
        # Default show=True for plotly (Jupyter doesn't always auto-render)
        plotly_kw.setdefault("show", True)
        # Map ax → fig for consistency with other convenience wrappers
        if "ax" in kwargs and "fig" not in plotly_kw:
            plotly_kw["fig"] = kwargs["ax"]
        return _plot_model_plotly(grillage_obj, **plotly_kw)
    elif backend == "matplotlib":
        mpl_kw = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "figsize",
                "ax",
                "members",
                "title",
                "show_nodes",
                "show_node_labels",
                "show_element_labels",
                "color_by_member",
                "show_supports",
                "show_rigid_links",
                "show",
            )
        }
        return _plot_model_matplotlib(grillage_obj, **mpl_kw)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'matplotlib' or 'plotly'.")


class PostProcessor:
    """Class to post-process the results from an of ospgrillage analysis result.

    As of version 0.2.0, ospgrillage compiles results using Xarray module.
    """

    def __init__(self, grillage, result):  # Union[xr.DataArray, xr.Dataset]
        """Init the class"""
        # store main vars
        self.grillage = grillage
        self.result = result

        # init vars
        self.shape_function_obj = ShapeFunction()

    def get_arbitrary_displacements(
        self, point: list, shape_function_type: str = "linear", component: str = "y"
    ):
        """Returns displacement (translation and rotational) from an arbitrary point by interpolation

        param point: list of coordinate. Default three elements [x,y=0,z]
        type point: list
        param component: Displacement component. Default "y"
        type component: str
        param shape_function_type: The shape function for interpolation. Default "linear"
        type shape_function_type: str
        """
        node_displacements = []
        node_coordinate = []

        # get the list of four nodes where arbitrary point lies
        nodes, grid_number = self.grillage._get_point_load_nodes(
            point=point
        )  # list of nodes
        if nodes is None:
            raise ValueError("Point is outside bridge mesh")

        # get results of each node of four nodes
        for node in nodes:
            node_displacements.append(
                self.result.sel(
                    Component=component,
                    Node=node,
                )
                .displacements.to_numpy()
                .tolist()[0]
            )
            node_coordinate.append(self.grillage.get_nodes(number=node))

        # interpolate for vertical displacement
        x = np.array([coord[0] for coord in node_coordinate])
        z = np.array([coord[2] for coord in node_coordinate])

        # get natural coordinate of point in grid
        eta, zeta = solve_zeta_eta(
            xp=point[0],
            zp=point[2],
            x1=x[0],
            z1=z[0],
            x2=x[1],
            z2=z[1],
            x3=x[2],
            z3=z[2],
            x4=x[3],
            z4=z[3],
        )
        if shape_function_type == "linear":
            shape_func = self.shape_function_obj.linear_shape_function(
                eta=eta, zeta=zeta
            )
        else:
            shape_func, _, _ = self.shape_function_obj.hermite_shape_function_2d(
                eta=eta, zeta=zeta
            )

        return sum([a * b for a, b, in zip(shape_func, node_displacements)])
