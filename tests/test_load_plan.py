"""Plan rendering checks inspect load locations, factors and clipped geometry."""

import copy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import ospgrillage as og


@pytest.fixture
def bridge():
    return og.create_grillage(
        bridge_name="plan",
        long_dim=20,
        width=7.2,
        skew=12,
        num_long_grid=5,
        num_trans_grid=5,
        edge_beam_dist=0.9,
        mesh_type="Oblique",
    )


def point(x, z, p=100):
    return og.create_load(point1=og.create_load_vertex(x=x, z=z, p=p))


def test_plan_clips_patch_and_wheels_to_skewed_deck_without_mutating(bridge, tmp_path):
    lane = og.create_load_case(name="Lane 1")
    lane.add_load(point(10, 2))
    lane.add_load(point(-100, 2))
    lane.add_load(
        og.create_load(
            **{
                f"point{i+1}": og.create_load_vertex(x=x, z=z, p=2)
                for i, (x, z) in enumerate([(-100, 1), (100, 1), (100, 3), (-100, 3)])
            }
        )
    )
    original = copy.deepcopy(lane.load_groups)
    fig, ax = plt.subplots()
    result = og.plot_load_plan(
        bridge, {lane: 0.8}, ax=ax, highlight=(10, 2), carriageway=(0.4, 6.8)
    )
    assert result is ax
    assert len(ax.collections) == 1  # off-deck wheel omitted
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[10, 2]])
    patch = ax.patches[0].get_path().vertices
    assert patch[:, 0].min() > -5 and patch[:, 0].max() < 25
    assert patch[:, 1].min() == pytest.approx(1)
    assert patch[:, 1].max() == pytest.approx(3)
    assert "0.8 × Lane 1" in ax.get_legend_handles_labels()[1]
    assert ax.get_aspect() == 1
    assert lane.load_groups[1]["load"].load_point_1 == original[1]["load"].load_point_1
    path = tmp_path / "plan.png"
    fig.savefig(path)
    assert path.stat().st_size > 10000
    plt.close(fig)


def test_named_increment_compound_and_stored_factors(bridge):
    compound = og.create_compound_load(name="vehicle")
    compound.add_load(point(5, 2))
    compound.add_load(point(7, 2))
    case = og.create_load_case(name="vehicle at position 3")
    case.add_load(compound, load_factor=1.3)
    bridge.moving_load_case_dict["vehicle"] = [
        {"name": case.name, "loadcase": case, "load_factor": 0.5}
    ]
    ax = og.plot_load_plan(bridge, {case.name: 1.6})
    assert len(ax.collections) == 2
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[7, 2]])
    assert "0.8 × vehicle at position 3" in ax.get_legend_handles_labels()[1]
    plt.close(ax.figure)


def test_named_static_and_combination(bridge):
    case = og.create_load_case(name="static")
    case.add_load(point(5, 2))
    bridge.add_load_case(case, load_factor=1.2)
    bridge.add_load_combination("ULS", {case.name: 1.5})
    for name, factor in (("static", 1.2), ("ULS", 1.5)):
        ax = og.plot_load_plan(bridge, name)
        assert f"{factor:g} × static" in ax.get_legend_handles_labels()[1]
        plt.close(ax.figure)


def test_line_and_nodal_loads(bridge):
    case = og.create_load_case(name="line and node")
    case.add_load(
        og.create_load(
            point1=og.create_load_vertex(x=-100, z=2, p=10),
            point2=og.create_load_vertex(x=100, z=2, p=10),
        )
    )
    tag = next(iter(bridge.Mesh_obj.node_spec))
    case.add_load(og.create_load(loadtype="nodal", node_tag=tag, Fy=30))
    ax = og.plot_load_plan(bridge, case)
    assert len(ax.collections) == 1
    lines = [line for line in ax.lines if line.get_linewidth() == 2]
    assert len(lines) == 1
    assert np.ptp(lines[0].get_xdata()) == pytest.approx(20)
    plt.close(ax.figure)


def test_mesh_only_and_zero_factor(bridge):
    case = og.create_load_case(name="zero")
    case.add_load(point(10, 2))
    for selection in (None, {case: 0}):
        ax = og.plot_load_plan(bridge, selection)
        assert len(ax.lines) > 0 and len(ax.collections) == 0
        plt.close(ax.figure)


@pytest.mark.parametrize("selection", ["missing", {"missing": float("nan")}, 42])
def test_invalid_case_selection_raises_before_drawing(bridge, selection):
    fig, ax = plt.subplots()
    with pytest.raises((ValueError, TypeError)):
        og.plot_load_plan(bridge, selection, ax=ax)
    assert not ax.lines
    plt.close(fig)


def test_does_not_overlay_alternative_moving_steps_in_a_combination(bridge):
    increments = []
    for i in range(2):
        case = og.create_load_case(name=f"step {i}")
        case.add_load(point(5 + i, 2))
        increments.append({"name": case.name, "loadcase": case, "load_factor": 1})
    bridge.moving_load_case_dict["vehicle"] = increments
    bridge.load_combination_dict["all steps"] = increments
    with pytest.raises(ValueError, match="alternative moving increments"):
        og.plot_load_plan(bridge, "all steps")


def test_clipping_preserves_a_hole_in_the_cell_union(bridge):
    from shapely.geometry import MultiPoint

    mesh = bridge.Mesh_obj
    interior = []
    for key, tags in mesh.grid_number_dict.items():
        polygon = MultiPoint(
            [np.asarray(mesh.node_spec[n]["coordinate"])[[0, 2]] for n in tags]
        ).convex_hull
        if 5 < polygon.centroid.x < 12 and 2 < polygon.centroid.y < 5:
            interior.append((key, polygon))
    key, polygon = interior[0]
    del mesh.grid_number_dict[key]
    case = og.create_load_case(name="hole")
    case.add_load(point(polygon.centroid.x, polygon.centroid.y))
    ax = og.plot_load_plan(bridge, case)
    assert len(ax.collections) == 0
    plt.close(ax.figure)
