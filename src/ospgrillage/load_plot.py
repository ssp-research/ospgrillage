"""Plan views of actual static or moving load arrangements."""

from collections.abc import Mapping
import math

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ospgrillage.load import CompoundLoad, LoadCase, NodalLoad

__all__ = ["plot_load_plan"]


def _case_records(model):
    records = list(model.load_case_list)
    for increments in model.moving_load_case_dict.values():
        records.extend(increments)
    return records


def _resolve_cases(model, selection):
    if selection is None:
        return []
    selections = (
        selection.items() if isinstance(selection, Mapping) else [(selection, 1.0)]
    )
    records = _case_records(model)
    result = []
    for key, multiplier in selections:
        if not math.isfinite(multiplier):
            raise ValueError("load factors must be finite")
        if isinstance(key, LoadCase):
            matches = [{"loadcase": key, "load_factor": 1.0}]
        elif isinstance(key, str):
            matches = [r for r in records if r["name"] == key]
            if not matches:
                matches = model.load_combination_dict.get(key, [])
                names = {r["name"] for r in matches}
                if any(
                    len(names.intersection(r["name"] for r in increments)) > 1
                    for increments in model.moving_load_case_dict.values()
                ):
                    raise ValueError(
                        "combination contains alternative moving increments; select a single increment per moving load"
                    )
            if not matches:
                raise ValueError(f"load case or combination {key!r} not found")
        else:
            raise TypeError("loadcase must contain case names or LoadCase objects")
        for record in matches:
            factor = multiplier * record.get("load_factor", 1.0)
            if not math.isfinite(factor):
                raise ValueError("stored load factors must be finite")
            result.append((record["loadcase"], factor))
    return result


def _loads(case, multiplier):
    def unpack(load, factor):
        if isinstance(load, CompoundLoad):
            for item in load.compound_load_obj_list:
                yield from unpack(item, factor)
        else:
            yield load, factor

    for group in case.load_groups:
        factor = multiplier * group.get("factor", 1.0)
        if not math.isfinite(factor):
            raise ValueError("per-load factors must be finite")
        yield from unpack(group["load"], factor)


def _parts(geometry):
    if geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _parts(part)
    else:
        yield geometry


def _polygon_patch(polygon, **kwargs):
    polygon = orient(polygon)
    rings = [polygon.exterior, *polygon.interiors]
    return PathPatch(
        Path.make_compound_path(*[Path(np.asarray(r.coords)) for r in rings]), **kwargs
    )


def plot_load_plan(
    grillage_obj,
    loadcase=None,
    *,
    ax=None,
    figsize=(11, 4),
    title=None,
    highlight=None,
    carriageway=None,
    show=False,
):
    """Plot a load arrangement over the actual mesh in the global x-z plane.

    Point/wheel and nodal loads are marked at their positions; line loads are
    drawn as lines and patches as shaded regions. Loads are clipped to the
    union of mesh cells, preserving skewed boundaries and gaps. No analysis or
    active OpenSees model is needed, and neither model nor loads are modified.

    Parameters
    ----------
    grillage_obj : OspGrillage
        Live model containing mesh geometry and load definitions.
    loadcase : str, LoadCase, mapping, or None
        Static case name, moving increment name, combination name, or a LoadCase
        object (which need not be registered on the model). A mapping from
        names/objects to factors overlays concurrent cases. Mapping factors
        multiply stored case factors and per-load factors. None draws the mesh.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; returned unchanged as an object.
    figsize : tuple, default (11, 4)
        Figure size in inches when creating axes.
    title : str, optional
        Plot title; defaults to "Load arrangement".
    highlight : tuple, optional
        Global (x, z) coordinates of a governing location, shown as a star.
        Convert girder-local station coordinates before passing them here.
    carriageway : tuple, optional
        Constant global (z_min, z_max) carriageway edges, clipped to the deck.
        Omit for models whose carriageway edges are not straight longitudinal lines.
    show : bool, default False
        Call matplotlib.pyplot.show after drawing. The axes are always returned.

    Returns
    -------
    matplotlib.axes.Axes
        Equal-aspect plan view, suitable for notebook display or file export.

    Notes
    -----
    This displays the selected arrangement; it does not find a governing case.
    Supply the witness from the response envelope or design check. Marker sizes
    and patch shading do not encode force magnitude or pressure variation.
    Legends identify cases and their effective case multipliers.
    Matplotlib is the rendering backend for this helper.

    Examples
    --------
    >>> ax = plot_load_plan(bridge, {"lane 1 position 3": 1., "lane 2 position 8": .8})  # doctest: +SKIP
    >>> ax.figure.savefig("governing_loads.png", bbox_inches="tight")  # doctest: +SKIP
    """
    from ospgrillage.postprocessing import _extract_mesh_data

    cases = _resolve_cases(grillage_obj, loadcase)
    mesh = grillage_obj.Mesh_obj
    nodes = {
        int(n): np.asarray(v["coordinate"], dtype=float)
        for n, v in mesh.node_spec.items()
    }
    cells = []
    for tags in mesh.grid_number_dict.values():
        coords = [
            nodes[int(n)][[0, 2]] for n in tags if n is not None and int(n) in nodes
        ]
        if len(coords) >= 3:
            cell = MultiPoint(coords).convex_hull
            if isinstance(cell, Polygon) and cell.area > 0:
                cells.append(cell)
    if not cells:
        raise ValueError("plot_load_plan requires a mesh with nondegenerate deck cells")
    deck = unary_union(cells)
    if highlight is not None and (
        len(highlight) != 2 or not np.isfinite(highlight).all()
    ):
        raise ValueError("highlight must be finite global (x, z) coordinates")
    if carriageway is not None and (
        len(carriageway) != 2
        or not np.isfinite(carriageway).all()
        or carriageway[0] >= carriageway[1]
    ):
        raise ValueError("carriageway must be increasing finite z coordinates")
    # Validate factors before touching caller-owned axes.
    expanded = [(case, factor, list(_loads(case, factor))) for case, factor in cases]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    data, _, _, _ = _extract_mesh_data(grillage_obj)
    seen = set()
    for entries in data.values():
        for ci, cj, tag, *_ in entries:
            if tag in seen:
                continue
            seen.add(tag)
            ax.plot([ci[0], cj[0]], [ci[2], cj[2]], color="0.75", lw=0.7, zorder=0)
    for part in _parts(deck.boundary):
        xy = np.asarray(part.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="0.25", lw=1.5, zorder=1)
    xmin, zmin, xmax, zmax = deck.bounds
    tolerance = max(xmax - xmin, zmax - zmin, 1) * 1e-9
    footprint = deck.buffer(tolerance)
    if carriageway is not None:
        for z in carriageway:
            for line in _parts(
                deck.intersection(LineString([(xmin - 1, z), (xmax + 1, z)]))
            ):
                if isinstance(line, LineString):
                    xy = np.asarray(line.coords)
                    ax.plot(xy[:, 0], xy[:, 1], "--", color="tab:blue", lw=1, zorder=2)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (case, factor, loads) in enumerate(expanded):
        colour = palette[i % len(palette)]
        drawn = False
        for load, effective in loads:
            if effective == 0:
                continue
            if isinstance(load, NodalLoad):
                if not any(
                    getattr(load, key) for key in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
                ):
                    continue
                tags = [load.node_tag] if np.isscalar(load.node_tag) else load.node_tag
                points = [nodes[int(n)][[0, 2]] for n in tags]
                geometry = MultiPoint(points)
            else:
                vertices = [getattr(load, f"load_point_{n}", None) for n in range(1, 9)]
                vertices = [v for v in vertices if v is not None]
                if not vertices or not any(v.p for v in vertices):
                    continue
                coords = [(v.x, v.z) for v in vertices]
                if len(coords) == 1:
                    geometry = Point(coords[0])
                elif len(coords) == 2:
                    geometry = LineString(coords)
                else:
                    geometry = Polygon(coords)
                    if not geometry.is_valid:
                        raise ValueError(f"invalid load patch in {case.name!r}")
            # Points on an edge tolerate rounding; areas/lines clip exactly.
            clipped = geometry.intersection(
                footprint if isinstance(geometry, (Point, MultiPoint)) else deck
            )
            for part in _parts(clipped):
                if isinstance(part, Point):
                    ax.scatter([part.x], [part.y], s=25, color=colour, zorder=4)
                elif isinstance(part, LineString):
                    xy = np.asarray(part.coords)
                    ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=2, zorder=3)
                elif isinstance(part, Polygon):
                    ax.add_patch(
                        _polygon_patch(
                            part,
                            facecolor=colour,
                            edgecolor=colour,
                            alpha=0.15,
                            zorder=1,
                        )
                    )
                else:
                    continue
                drawn = True
        if drawn:
            ax.plot([], [], "o", color=colour, label=f"{factor:g} × {case.name}")
    if highlight is not None:
        ax.plot(
            *highlight, "*", ms=14, color="black", label="Governing location", zorder=5
        )
    ax.set(
        xlabel="Global longitudinal x",
        ylabel="Global transverse z",
        title=title or "Load arrangement",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin - 0.03 * max(xmax - xmin, 1), xmax + 0.03 * max(xmax - xmin, 1))
    ax.set_ylim(zmin - 0.08 * max(zmax - zmin, 1), zmax + 0.08 * max(zmax - zmin, 1))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    if show:
        plt.show()
    return ax
