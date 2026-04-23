# Two-FET Shared Diffusion Skill

Use this skill when the request is about a two-transistor hello world with merged or shared diffusion.

Recommended primitive:

- `glayout.placement.two_transistor_interdigitized.two_transistor_interdigitized`

Why this is the right MVP:

- It already exists in the repo and encodes a stable two-device placement pattern.
- The topology choice stays in the Python source, which matches the desired separation between runtime sizing parameters and hard-coded placement/routing.
- It is a good first target for self-repair because success is easy to validate: the file should compile, run, and write a GDS.

Expected output shape:

- A single Python file
- One `build_*` function exposing sizing parameters such as `width`, `length`, and `fingers`
- Placement and routing choices fixed inside the function
- A CLI entry point that writes a GDS with `--output-gds`
