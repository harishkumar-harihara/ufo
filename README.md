## Tuning parallel back-projection algorithm for modern GPU architectures

### GPU for Tomographic Reconstruction
X-Ray imaging uses high speed cameras that acquire data at a higher rate i.e hundreds of thousands of frames per seconds. The reconstruction of these acquired images takes a longer time. So, in general, an offline reconstruction is performed. But with the aid of GPUs, the reconstruction rate can be improved and is possible to study the acquired data in real time.

### Filtered Back-Projection (FBP)
FBP algorithm is a simple reconstruction technique, considered in the scope of this project. The reconstructed value of a pixel is calculated by integrating the projection value of 2D cross-sectional slice for all projection angles. The method relies on texture engine, and its efficiency can be further improved.

### Goals 
In [1] Chilingaryan et al, proposed an optimal algorithm over texture engine. The proposed approach involves reconstructing multiple slices in parallel and to adapt GPU friendly cache placement strategies.

### Proposed approach
The algorithm involves three kernels:

**Interleave kernel**
Transforms the cross-sectional slices into vectors, and of certain data type. The data is transformed into 32-bit, 16-bit and 8-bit data type. Depending on the data type, the number of slices to be reconstructed is decided. For example, if the data is single precision (32-bit), 2 slices are reconstructed in parallel. For other two precision modes, 4 slices are reconstructed. The texture engine operates in full speed for 64-bit data. On reconstructing two slices, first slice value is stored in x component and second in y component of float2 vector.

**Backprojection kernel**
First the elements of 2x2 square are arrange in Z-curve mapping for improved locality of fetches. Then compute the partial sums, in such a way that 4 threads are responsible to compute the projection values of a pixel, i.e each thread computes over a quarter of available projections. Finally, transfer the partial sum to shared memory and perform a standard reduction.
  
**Uninterleave kernel**
Write back the vectors to corresponding cross-sectional slices and handle necessary data type conversions.

### Results

Data size: 4.3 GB

Experiment | Execution time (s) |	Texture fill rate (GTexels/s)	| Throughput (GB/s)
-----------|--------------------|-------------------------------|------------------
Standard |	5.948386 |	184.8420106 |	0.7220391037
Single precision |	2.303378	| 238.6737278	| 1.864638499
Half precision |	1.174175 |	234.1030144	 | 3.6578596
8-bit integer |	1.158446	| 237.2815884	| 3.707524819


### Reference
[1] S. Chilingaryan, E. Ametova, A. Kopmann, and A. Mirone, “Balancing load of gpu subsystems to accelerate image reconstruction in parallel beam tomography,” in 2018 30th International Symposium on Computer Architecture and High Performance Computing (SBAC-PAD), Sep.2018, pp. 158–166.

### License

Both ufo-core and ufo-filters are licensed under LGPL 3.


## Citation

If you use this software for publishing your data, we kindly ask to cite the article below.

Vogelgesang, Matthias, et al. "Real-time image-content-based beamline control for smart 4D X-ray imaging." Journal of synchrotron radiation 23.5 (2016): 1254-1263.
