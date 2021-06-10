## Tuning parallel back-projection algorithm for modern GPU architectures

### Description
Filtered back-projection algorithm is traditionally used for tomographic reconstruction. Chilingaryan et al.[1] proposed how to implement the algorithm efficiently on modern GPU architectures.

### Task
To implement the optimized algorithm relying on texture engine to perform reconstruction i.e.to implement Algorithm 1 in [1].

### Problem
A 3D reconstruction problem can be split into a stack of 2D reconstructions performed with cross-sectional slices. To reconstruct a slice, the projection values are smeared back over the 2D cross section and integrated over all the projection angles. In the older variant of ufo-backproject, this reconstruction is done slice by slice. The performance of this approach is dominated by throughput of texture engine but rather keeps all other components of GPU architecture under-utilized.

### Solution
Implement the algorithm to process an arbitrary number of sinograms instead of processing one slice at a time. 

If the number of sinograms is more than 4, then process 4 slices at a time in three kernels:

* Interleave kernel
  The kernel is adjusted to use the float4 type, to write the first slice to x component, second to y, third to z and fourth to w component respectively.

* Backprojection kernel
  Two distinct processing stages are executed. First the partial sums are computed in an 4-element array. The outer loop starts from the ﬁrst projection assigned   to the current thread and steps over the projections which are processed in parallel. At each iteration the constants are loaded and inner loop is executed to     process 4 pixels.
  
* Uninterleave kernel
  Write the 4-element reconstructed array to 4 slices.

When the number of sinograms is not a multiple of 4, the remaining slices are processed using the older variant kernel but without iteration of slice at host.
For example, to process 15 sinograms at each iteration, first 12 slices are processed using three kernels: interleave-backprojection-uninterleave (process 4-slices) and then remaining 3 slices are processed using backprojection kernel (process slice-wise).

### Results
stack size | older variant | with loop in host | no loop
-----------|---------------|-------------------|--------
1 |	5.960049 |	0	| 0
2 |	0	| 5.795506 | 8.886319
4	| 0	| 4.597504 | 2.264165
8 |	0	| 8.695537 | 2.281061
16 |	0 |	16.20616 | 2.338949

### License

Both ufo-core and ufo-filters are licensed under LGPL 3.


## Citation

If you use this software for publishing your data, we kindly ask to cite the article below.

Vogelgesang, Matthias, et al. "Real-time image-content-based beamline control for smart 4D X-ray imaging." Journal of synchrotron radiation 23.5 (2016): 1254-1263.
