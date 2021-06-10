## Tuning parallel back-projection algorithm for modern GPU architectures

### Description
Filtered back-projection algorithm is traditionally used for tomographic reconstruction. Chilingaryan et al.[1] proposed how to implement the algorithm efficiently on modern GPU architectures.

### Task
To implement the optimized algorithm relying on texture engine to perform reconstruction i.e.to implement Algorithm 1 in [1].

A 3D reconstruction problem can be split into a stack of 2D reconstructions performed with cross-sectional slices. To reconstruct a slice, the projection values are smeared back over the 2D cross section and integrated over all the projection angles. In the older variant of ufo-backproject, this reconstruction is done slice by slice. The performance of this approach is dominated by throughput of texture engine but rather keeps all other components of GPU architecture under-utilized.


### License

Both ufo-core and ufo-filters are licensed under LGPL 3.


## Citation

If you use this software for publishing your data, we kindly ask to cite the article below.

Vogelgesang, Matthias, et al. "Real-time image-content-based beamline control for smart 4D X-ray imaging." Journal of synchrotron radiation 23.5 (2016): 1254-1263.
