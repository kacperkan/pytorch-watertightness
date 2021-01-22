# Watertigthness measure

## Description
The program allows you to measure watertigthness of measure. While the 
watertigthness is usually determined as a boolean value (an object is or 
isn't watertight), we wanted to obtain a measure that would tell us how many
casted rays would fall out of the mesh. It is done as following:

1. Randomly sample `count` points on a surface of the mesh.
2. Add a buffer multiplied by a small value `eps` to avoid problems due
to rounding of float.
3. Get normal vectors of faces associated with each point for each point. It 
is often available out-of-the-box in `trimesh` library. Negate these normals
(now we have `directions`).
4. Once we have points and normals directed towards the mesh, we calculate a
parity - how many times a ray casted from the point along its direction passes
any triangle. If the number is even, then we say that the ray passed parity
test and the mesh is watertight from the perspective of that ray
5. Perform the same action for each ray.
6. Calcuate ratio between the count of rays that passed the test and total rays
casted (equal to `count`).


## Requirements

- GPU :) The processing was not adapted for CPU. You can perform similar calculations
using `trimesh` library.
- `torch>=1.4.0` (`==1.4.0` was validated).
- `python==3.7.0` (it may work on different versions but you have to adapt
 `Makefile` in that case)

## Installation
```
make -j<cpu-count>
```

This will produce `watertightness` directory that is available in the directory 
one above project's directory. Eg.
```
$ cd /Projects/pytorch-watertightness
$ make -j<cpu-count>
$ cd ../watertightness
```

## Caveat
Copying in the `Makefile` is done explicitly since we know the output directory
of the binary. However, it may differ, depending on your python version. In 
such a case, the compilation will crush and you need change lines 80, 81:
```
@ mv build/lib.linux-x86_64-<version>/watertightness ..
@ mv build/lib.linux-x86_64-<version>/*.so ../watertightness/
```
where `<version>` is the version of your python environment.


## Usage
If everything works, you are ready to go! You can take a look how to use the program
inside the `metric.py` file.

```
>>> import waterightness
>>> watertigthness.run_example()
```

This will create two icospheres where half of faces in one of the is removed. In
the second one, the watertightness measure should be equal almost 0.

```
>>> import watertigthness
>>> import trimesh
>>> mesh: trimesh.Trimesh = ...
>>> watertightness.calc_watertigthness_trimesh(mesh, count=...)
```

If you know all necessary parameters (points, directions and triangles) and have
them in `torch.Tensor` format (for example, you could have obtained them 
with `pytorch3d` library) then you can use the following:

```
>>> import watertigthness
>>> import torch
>>> triangles: torch.Tensor = ...
>>> origins: torch.Tensor = ...
>>> directions: torch.Tensor = ...
>>> watertightness.calc_watertigthness(origins, directions, triangles)
```

## License 
MIT License @ 2020 Kacper Kania