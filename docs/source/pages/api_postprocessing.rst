Post-processing
===============

Influence-line and influence-surface post-processing preserves the
`shape_function` metadata from the stored influence study. When a study was run
with `shape_function="hermite"`, that corresponds to the higher-order load
distribution path: Hermite on quadrilateral regions and the DKT-style condensed
distributor on three-node triangular regions.

For influence surfaces, :func:`~ospgrillage.postprocessing.plot_is` supports
``coordinate_space="station"`` and ``coordinate_space="physical"``. Physical
space uses mapped deck coordinates and triangulated rendering so curved/skewed
decks are plotted as contiguous surfaces (Matplotlib and Plotly backends).

For influence lines, :func:`~ospgrillage.postprocessing.plot_il` supports both
``view="ordinate"`` and ``view="path"``. Path view overlays IL ordinates in
3D along the load trajectory using model geometry embedded in the results file.

Use :func:`~ospgrillage.postprocessing.plot_load_plan` to draw a selected static
case, moving increment or concurrent arrangement over the actual deck mesh.
Wheel markers, line loads and shaded patches are clipped to the deck boundary.
The function accepts a mapping of case names or LoadCase objects to factors and
an optional governing-location marker. It renders a supplied witness rather
than performing a governing-load search. See the executable
:doc:`../notebooks/load_arrangements` tutorial.

Factory functions
-----------------

.. autosummary::
   :toctree: generated/

   ~ospgrillage.postprocessing.create_envelope
   ~ospgrillage.postprocessing.create_influence_line
   ~ospgrillage.postprocessing.create_influence_surface
   ~ospgrillage.postprocessing.plot_il
   ~ospgrillage.postprocessing.plot_is
   ~ospgrillage.postprocessing.plot_force
   ~ospgrillage.postprocessing.plot_bmd
   ~ospgrillage.postprocessing.plot_sfd
   ~ospgrillage.postprocessing.plot_tmd
   ~ospgrillage.postprocessing.plot_def
   ~ospgrillage.postprocessing.plot_model
   ~ospgrillage.postprocessing.plot_srf
   ~ospgrillage.postprocessing.plot_load_plan
   ~ospgrillage.postprocessing.model_proxy_from_results

Class reference
---------------

Envelope
~~~~~~~~

.. autoclass:: ospgrillage.postprocessing.Envelope
   :show-inheritance:

InfluenceLine
~~~~~~~~~~~~~

.. autoclass:: ospgrillage.postprocessing.InfluenceLine
   :show-inheritance:

InfluenceSurface
~~~~~~~~~~~~~~~~

.. autoclass:: ospgrillage.postprocessing.InfluenceSurface
   :show-inheritance:

PostProcessor
~~~~~~~~~~~~~

.. autoclass:: ospgrillage.postprocessing.PostProcessor
   :show-inheritance:
