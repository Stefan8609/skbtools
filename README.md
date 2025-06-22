# skbtools

Utilities for acoustic ray tracing, geometry processing and simple plotting.

This project bundles several standalone modules used in GPS / acoustic
localization experiments. The code lives under `src/` and is structured as a
Python package.

## Installation

Clone the repository and install in editable mode to expose the `geometry`,
`acoustics` and other subpackages:

```bash
pip install -e .
```

This will also install the numerical dependencies defined in
`pyproject.toml` (NumPy, SciPy, pandas, matplotlib, numba and pymap3d).

If working straight from the source you can alternatively add the `src`
directory to your `PYTHONPATH`.

## Package overview

### geometry

Low level helpers for working with point clouds and rigid-body motion.
The main functions are

- **rotationMatrix(angle, vect)** – compute a Rodrigues rotation matrix
  for a unit vector.
- **projectToPlane(vect, normal)** – project a vector onto a plane.
- **fitPlane(xs, ys, zs)** – fit the best plane through a set of points.
- **initializeFunction** and **findXyzt** – describe and reconstruct a
  point relative to a reference cloud.
- **findRotationAndDisplacement** – estimate the rotation matrix and
  translation between two point clouds.

### acoustics

Tools for simple sound velocity modelling and ray tracing.

- **ray_tracing** – trace an acoustic ray through a stratified velocity
  profile.
- **ray_trace_locate** – compute the launch angle that yields a desired
  range.
- **depth_to_pressure_Leroy** and **depth_to_pressure** – convert depth
  and latitude to pressure.
- **DelGrosso_SV**, **UNESCO_SV**, **NPL_ESV**, **Mackenzie_ESV** and
  **Coppens_ESV** – various empirical sound‑speed equations.

### plotting

Convenience helpers for visualising results.

- **plotPlane(point, normal, xrange, yrange)** – create a 3‑D plot of a
  plane.
- **printTable(headers, data)** – render a simple text table.

### examples and GeigerMethod

The `examples` and `GeigerMethod` directories contain scripts and
notebooks used during development. They illustrate typical workflows for
processing GPS data and analysing transducer positions.

## Usage

After installation you can import the modules directly. A small example
illustrating a few features:

```python
import numpy as np
from geometry.rodrigues import rotationMatrix
from acoustics.ray_tracing import ray_tracing

# Rotation about the z-axis
R = rotationMatrix(np.pi/2, [0, 0, 1])

# Ray tracing through a constant sound speed profile
depth = np.linspace(0, 1000, 101)
cz = np.full_like(depth, 1500.0)
x, dz, t = ray_tracing(45.0, 0.0, 1000.0, depth, cz)
print(x, t)
```

See the files in `tests/` for more examples of calling individual
functions.
