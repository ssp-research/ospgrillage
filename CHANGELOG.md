# Changelog

All notable changes to *ospgrillage* are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `plot_load_plan()` renders static cases, selected moving increments and
  concurrent load arrangements over the deck mesh. Point, compound, nodal,
  line and patch loads are supported, with skew/hole-aware clipping, optional
  carriageway edges and a governing-location marker. Includes an executable
  notebook and accepts existing Matplotlib axes without modifying the model.
- Shapely dependency for clipping loads to the union of actual deck cells.

### Fixed
- Correct M1600 inter-group axle spacings to AS 5100.2:2017 Figure 7.2.4.
  The variable middle gap and final 5 m gap are measured from the preceding
  group's last axle; previously both were shortened by 2.5 m in SI output.
  The minimum-gap vehicle now spans 25 m instead of 20 m. Previously generated
  M1600 load effects will change and should be recalculated.
- Convert the M1600 variable gap and transverse wheel track with the remaining
  geometry for imperial output.

### Changed
- M1600 `gap` defaults to 6.25 m and rejects non-finite or smaller values.
  It specifies the clear distance in metres, including for imperial output;
  larger gaps remain available for adverse-load searches. The generator creates
  axle loads only; users must add the lane UDL and applicable design factors.

---

## [0.5.0] — 2026-03-16

### Added
- **Interactive Plotly backend** for `plot_bmd()`, `plot_sfd()`, and `plot_def()`:
  pass `backend="plotly"` for 3D rotation, zoom, and hover in Jupyter notebooks
  and browser windows. Plotly is bundled in the `gui` extra
  (`pip install ospgrillage[gui]`).
- **`LoadVertex`** namedtuple — the preferred name for the coordinate+magnitude
  tuple that defines where a load acts and how much.  `LoadPoint` is retained as
  a backwards-compatible alias.
- `GrillageMember.section` and `.material` read-only properties.
- `Mesh.orthogonal` attribute (was missing; code paths that checked it would
  raise `AttributeError` on oblique meshes).
- `beam_z_spacing` → `beam_width` deprecation shim in `create_grillage()`: the
  old name now emits `DeprecationWarning` and maps correctly instead of being
  silently ignored.
- `Section(E=…)` / `Section(G=…)` now raise `ValueError` with a clear message
  (elastic moduli belong on `Material`, not `Section`).
- **Plotting keyword arguments** for all plotting functions (`plot_force`,
  `plot_defo`, `plot_bmd`, `plot_sfd`, `plot_def`): `figsize`, `ax` (existing
  matplotlib Axes), `scale`, `title`, `color`, `fill`, `alpha`, and `show`.
  Plotly equivalents (`fig`, `figsize`, `scale`, `title`, `alpha`) are forwarded
  through convenience wrappers.
- **`plot_model()`** function for visualising grillage mesh geometry with
  `backend="matplotlib"` (2-D plan view) and `backend="plotly"` (interactive 3-D).
  Replaces `og.opsv.plot_model()` and `og.opsplt.plot_model()`.
- 5 new Plotly tests, 8 new plotting-kwargs tests, 5 new plot_model tests, plus
  tests for `beam_z_spacing`, `Section(E=)`, `GrillageMember` properties, and
  `Mesh.orthogonal`.

### Changed
- **`vfo` is no longer a required dependency.** It is still accessible via
  `og.opsplt` (with a deprecation warning) if installed, but `pip install
  ospgrillage` no longer pulls it in. Use `og.plot_model()` instead.
- `og.opsv` and `og.opsplt` re-exports now emit `DeprecationWarning`. Use
  `og.plot_model()` for mesh visualisation.
- `plot_deflection` renamed to `plot_def` (consistent with `plot_bmd` / `plot_sfd`).
- Documentation restructured: `performing_analysis.md` split into
  `defining_loads.md` (load types, compound loads, load cases, moving loads) and
  the "Running analysis" section merged into `getting_results.md` (now titled
  "Analysis and results").
- All docs code examples updated to use `LoadVertex` instead of `LoadPoint`.
- 15+ documentation content fixes: removed non-existent `set_material()` reference,
  corrected `link_nodes_width` → `beam_width`, `NodalForce` → `NodeForces`,
  `type=` → `loadtype=`, duplicate member assignments, missing Sphinx
  cross-references, and sparse pages.
- Added missing methods to API reference: `set_spring_support`,
  `set_previous_state`, `store_state`, `parse_moving_load_cases`.

### Fixed
- GUI code generation: `set_member()` calls wrote `member=interior_main_beam`
  (unquoted) instead of `member="interior_main_beam"`.
- Removed stale `html_static_path` and `html_theme_path` from Sphinx `conf.py`
  (referenced non-existent `_static/` and `_themes/` directories, causing build
  warnings).

### Deprecated
- `LoadPoint` — use `LoadVertex` instead.  `LoadPoint` remains as an alias and
  existing code is unaffected.

---

## [0.4.1] — 2026-03-09

### Added
- 30 new tests across `test_load.py`, `test_material.py`, `test_member.py`, and the
  new `test_ospgui.py`; overall coverage rises from 71 % to 75 %.
- `test_ospgui.py`: new test module verifying the PyQt5-absent import guard and
  graceful `main()` exit.

### Changed
- Minimum supported Python version raised from 3.9 to 3.10, allowing use of
  PEP 604 `X | Y` union-type syntax throughout the codebase.
  **Note:** Python 3.9 reached end-of-life in October 2025.
- Sphinx documentation source converted from reStructuredText to Markdown using
  MyST-Parser; `myst-parser` added to the docs-build dependency list.
- `_OpsProxy` dual-mode dispatch pattern: a thin proxy around OpenSeesPy that either
  executes commands live or serialises them to a `.tcl`/`.py` script file, controlled
  by a single `py_file` flag on `create_grillage`.
- `_OpsProxy._dispatch(call)` helper that routes pre-built `(name, args, kwargs)`
  tuples through the proxy, eliminating all `eval()` in the analysis path.
- Load assignment pipeline refactored from format-string building to
  `(func_name, args, kwargs)` tuples throughout `_assign_load_to_four_node()`,
  `_distribute_load_types_to_model()`, `Analysis._time_series_command()`, and
  `Analysis._pattern_command()`; `evaluate_analysis()` dispatches via `_dispatch()`.
- `Envelope.get()` now uses `getattr(da, self.selected_xarray_command)(dim="Loadcase")`
  instead of `eval()`; the dead format-string infrastructure in `Envelope.__init__`
  has been removed.
- `ospgui.py` log output converted from `print()` to `logging.getLogger(__name__)`.
- NumPy-style docstrings added to all public functions and classes across every
  source module (`osp_grillage.py`, `load.py`, `mesh.py`, `members.py`, `utils.py`,
  `postprocessing.py`).
- `sphinx_autodoc_typehints` added to Sphinx extension list so type annotations are
  rendered automatically in the HTML docs.

### Removed
- Nine dead command-string attributes from `Analysis.__init__` that became obsolete
  after the proxy refactor (`wipe_command`, `numberer_command`, `system_command`,
  `constraint_command`, `algorithm_command`, `analyze_command`, `analysis_command`,
  `intergrator_command`, `sensitivity_integrator_command`, `remove_pattern_command`).
- Dead `analysis_arguments` dict from `Analysis.__init__`.
- Unused `scipy.interpolate` imports from `postprocessing.py`.
- Five orphaned RST stub pages that referenced non-existent autodoc targets
  (`Analysis.rst`, `Material.rst`, `Loads.rst`, `GrillageMember.rst`, `OpsGrillage.rst`).

### Fixed
- `Loads.get_magnitude()` crashed with `AttributeError` on `None` items in
  `point_list`; now skips undefined load points.
- `ospgui` crashed on import with `ModuleNotFoundError: No module named 'PyQt5'`
  in environments without the optional GUI dependency; now exits gracefully with
  an actionable error message.
- `_OpsProxy` `str | None` annotation caused `TypeError` on Python ≤ 3.9 at import
  time (resolved by the Python 3.10 minimum bump).
- Compound `NodalLoad` bug in `_distribute_load_types_to_model()` where
  `load_str += string` was spreading individual characters into the list; now
  correctly appends a single tuple.
- `PatchLoading` cyclic-rotation-aware vertex validation via `_is_cyclic_rotation()`
  helper: any valid starting point of the CCW polygon is now accepted.
- `NodalLoad.get_nodal_load_call()` returning a structured tuple instead of a raw
  command string.
- `stitch_slab_x_spacing` typo corrected throughout `ospgui.py` (was `stich`).
- Docs build workflow now installs `myst-parser`; `sphinx_docs_to_gh_pages.yml`
  modernised to use `peaceiris/actions-gh-pages@v4`.

### Docs
- Documentation navigation restructured into four top-level sections: Getting
  Started, User Guide, API Reference, and Additional Resources.
- API reference split into per-module pages (Grillage, Material, Members, Load,
  Mesh, PostProcessing); Load module further subdivided into load types, load
  cases, and moving loads.
- All documentation source files renamed to match their page titles
  (e.g. `Module_description.md` → `creating_grillage_models.md`); source folder
  renamed `rst/` → `pages/`.
- Pandoc conversion artefacts removed throughout: escaped characters in prose
  (`\'`, `\"`, `\_`), malformed MyST/RST directives, broken internal anchors,
  and Pandoc grid tables converted to GFM pipe tables.
- Docstrings improved for `PatchLoading`, `LoadCase`, `CompoundLoad`, and
  `OspGrillage` (full `:param:` / `:raises:` fields; NumPy `Attributes` section
  rewritten as plain prose to avoid raw-text rendering without Napoleon).
- *Getting Results* page rewritten with an xarray concept overview, a summary
  table of data variables, and annotated `.sel()` / `.isel()` examples.
- Contributing guidelines page added via MyST `{include}` of root
  `CONTRIBUTING.md` (single source of truth).
- JOSS citation added to the front page and Additional Resources.
- Jupyter example notebooks cleaned up: version-output cells and trailing empty
  cells removed from all four notebooks.

---

## [0.4.0] — 2024-08-06

### Added
- GUI-based geometry generator (`ospgui`) for interactive model creation without
  writing Python code ([#128](https://github.com/MonashSmartStructures/ospgrillage/pull/128)).

### Fixed
- NumPy 2 and upstream dependency compatibility issues
  ([#126](https://github.com/MonashSmartStructures/ospgrillage/pull/126)).
- GitHub Pages deployment workflow
  ([#109](https://github.com/MonashSmartStructures/ospgrillage/pull/109)).

---

## [0.3.2] — 2023-10-20

### Changed
- Replaced deprecated `openseespyvis` visualisation back-end with `vfo`
  ([#74](https://github.com/MonashSmartStructures/ospgrillage/pull/74)).
- Mass and displacement interpolator updated
  ([#72](https://github.com/MonashSmartStructures/ospgrillage/pull/72)).
- Code reformatted with `black`
  ([#73](https://github.com/MonashSmartStructures/ospgrillage/pull/73)).

### Fixed
- Plot module errors following the `vfo` migration
  ([#107](https://github.com/MonashSmartStructures/ospgrillage/pull/107)).

---

## [0.3.1] — 2023-01-04

### Changed
- Package metadata migrated from `setup.cfg` to PEP 621 `pyproject.toml`.
- Documentation build fixed for `src`-layout packages.

---

## [0.3.0] — 2022-11-14

### Added
- Multi-span orthogonal meshing feature.
- Refined member assignment: `set_member()` now supports per-group overrides for
  common grillage elements with multiple groups.
- Rotational spring support via a dedicated `Material` type backed by OpenSeesPy's
  `zeroLength` element.

---

## [0.2.1] — 2022-04-10

### Fixed
- Minor paper / citation tweaks; dependency version pins.

---

## [0.2.0] — 2022-03-xx

### Added
- Multi-span meshing: intermediate edge construction lines and stitch elements
  between spans.
- Curve-mesh support: sweep path can now follow a curved line.
- Custom transverse member spacings for oblique meshes.
- Example Jupyter notebooks restructured under `docs/source/notebooks/`.

---

## [0.1.1] — 2022-02-xx

### Fixed
- Miscellaneous bug fixes and documentation corrections following initial release.

---

## [0.1.0] — 2021-11-xx

### Added
- Initial public release.
- Beam-only, beam-with-rigid-links, and shell-beam hybrid model types.
- Orthogonal and oblique meshing algorithms.
- Full load suite: `PointLoad`, `LineLoading`, `PatchLoading`, `NodalLoad`,
  `CompoundLoad`, `MovingLoad`.
- `LoadCase`, `LoadModel`, and `Path` helpers.
- `Envelope` and `PostProcessor` post-processing utilities.
- Sphinx documentation published to GitHub Pages.
