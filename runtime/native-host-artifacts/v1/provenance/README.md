# NextGenDA certified native host runtime

This bundle contains the native host artifacts required by the NextGenDA
sequential ensemble/data-assimilation runtime.

The `ngen` executable is a modified build of CIROH-UA/ngen at commit
`fab1cd54084a067f9a6b8d77dfbbbbd5cdea4b8a`.

NextGenDA modifications add the version-2 step-hook host API used for
same-process model-state access and state transfer. The modified upstream
files are:

- `CMakeLists.txt`
- `include/realizations/catchment/Bmi_Multi_Formulation.hpp`
- `include/realizations/catchment/Bmi_Module_Formulation.hpp`
- `src/NGen.cpp`
- added `include/ngiab_da_step_hook_api_v2.h`

The complete source patch is retained in
`provenance/ngiab-da-step-hook-v2.patch`.

The NGen upstream license and disclaimer are retained under
`third_party/ngen/`.

These artifacts do not change hydrologic model equations, parameter values,
the routing analysis equations, or the data-assimilation algorithms. They
provide the native host interface required for state access at runtime.
