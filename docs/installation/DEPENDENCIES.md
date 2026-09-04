# NextGenDA Dependencies

The final public installation instructions are validated from a clean
GitHub clone before release.

## Python

Supported release-candidate environment:

- Python 3.12

Python packages:

- NumPy 1.26.4
- pandas 3.0.5
- SciPy 1.17.1
- xarray 2026.7.0
- netCDF4 1.7.4
- h5py 3.16.0
- PyYAML 6.0.3

The exact environment is provided by:

- `requirements.txt`
- `environment.yml`

## Docker

Production NextGenDA execution uses Docker containers.

Docker must therefore be installed and running.

## NextGen / ngen

Production simulations require the supported NextGen/ngen runtime.

The final beginner guide will include:

1. acquisition,
2. required build tools,
3. supported revision,
4. build procedure,
5. verification.

## t-route

NextGenDA uses t-route for routing and routing data assimilation.

The following Python namespaces originate from the supported t-route
installation/source tree:

- `troute`
- `nwm_routing`
- `bmi_troute`

They are therefore intentionally not installed from `requirements.txt`.

## Installation verification

After installation:

```bash
python scripts/check_prerequisites.py

The final public guide will contain complete commands for WSL2/Linux users.
