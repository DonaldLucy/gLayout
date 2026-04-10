# GLADE

GLADE is the research layer built on top of gLayout.

The goal is to turn parameterized layout generators into a benchmarkable design
space, then learn quality oracles and search policies over that space.

The implementation is intentionally staged:

1. Start with a single generator (`current_mirror`)
2. Build a structured dataset from exhaustive parameter sweeps
3. Train feasibility / quality surrogate models
4. Add guided search over the generator design space
5. Extend to larger cell families and hierarchical composition

Directory overview:

- `common/`: shared environment, parsing, and serialization helpers
- `current_mirror/`: parameter space and dataset collection for the first MVP
- `models/`: placeholders for surrogate training code
- `search/`: placeholders for active exploration algorithms

This layout keeps the research layer separate from `src/glayout/`, which stays
focused on the generator library itself.

Bare-metal note:

- The collector can run without the Chipathon container if the host already has
  `magic`, `netgen`, and `ngspice` on `PATH`.
- Set `PDK_ROOT` to a valid SKY130 installation before running the collector.
- A minimal Python environment only needs `pip install -e .`; training extras
  can be installed separately later.
