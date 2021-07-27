/*
 * Copyright (C) 2011-2013 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */


constant sampler_t volumeSampler = CLK_NORMALIZED_COORDS_FALSE |
                                   CLK_ADDRESS_CLAMP_TO_EDGE |
                                   CLK_FILTER_NEAREST ;


kernel void
backproject_nearest (global float *sinogram,
                     global float *slice,
                     constant float *sin_lut,
                     constant float *cos_lut,
                     const unsigned int x_offset,
                     const unsigned int y_offset,
                     const unsigned int angle_offset,
                     const unsigned n_projections,
                     const float axis_pos)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int width = get_global_size(0);
    const float bx = idx - axis_pos + x_offset + 0.5f;
    const float by = idy - axis_pos + y_offset + 0.5f;
    float sum = 0.0f;

    for(int proj = 0; proj < n_projections; proj++) {
        float h = axis_pos + bx * cos_lut[angle_offset + proj] + by * sin_lut[angle_offset + proj];
        sum += sinogram[(int)(proj * width + h)];
    }

    slice[idy * width + idx] = sum * M_PI_F / n_projections;
}

kernel void
interleave_float4 (global float *sinogram,
            write_only image2d_array_t interleaved_sinograms)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    int sinogram_offset = idz*4;

   write_imagef(interleaved_sinograms, (int4)(idx, idy, idz, 0),
                 (float4)(sinogram[idx + idy * sizey + (sinogram_offset) * sizex * sizey],
                          sinogram[idx + idy * sizey + (sinogram_offset + 1) * sizex * sizey],
                          sinogram[idx + idy * sizey + (sinogram_offset + 2) * sizex * sizey],
                          sinogram[idx + idy * sizey + (sinogram_offset + 3) * sizex * sizey]));

}

kernel void
interleave_float2 (global float *sinogram,
                    write_only image2d_array_t interleaved_sinograms)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    int sinogram_offset = idz*2;

    // At each pixel, pack 2 slices in Z-projection
    write_imagef(interleaved_sinograms, (int4)(idx, idy, idz, 0),
                    (float4)(sinogram[idx + idy * sizey + (sinogram_offset) * sizex * sizey],
                                      sinogram[idx + idy * sizey + (sinogram_offset + 1) * sizex * sizey],0.0,0.0));
}

kernel void
interleave_uint (global uint *sinogram,
            write_only image2d_array_t interleaved_sinograms)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    int sinogram_offset = idz*4;

   write_imageui(interleaved_sinograms, (int4)(idx, idy, idz, 0),
                 (uint4)(sinogram[idx + idy * sizey + (sinogram_offset) * sizex * sizey],
                          sinogram[idx + idy * sizey + (sinogram_offset + 1) * sizex * sizey],
                          sinogram[idx + idy * sizey + (sinogram_offset + 2) * sizex * sizey],
                          sinogram[idx + idy * sizey + (sinogram_offset + 3) * sizex * sizey]));

}

kernel void
uninterleave_float4 (  global float4 *reconstructed_buffer,
                global float *output)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);
    int output_offset = idz*4;

    output[idx + idy*sizey + (output_offset)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].x);
    output[idx + idy*sizey + (output_offset+1)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].y);
    output[idx + idy*sizey + (output_offset+2)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].z);
    output[idx + idy*sizey + (output_offset+3)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].w);
}

kernel void
uninterleave_float2 (global float2 *reconstructed_buffer,
                      global float *output)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);
    int output_offset = idz*2;

    output[idx + idy*sizey + (output_offset)*sizex*sizey] = reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].x;
    output[idx + idy*sizey + (output_offset+1)*sizex*sizey] = reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].y;
}

kernel void
uninterleave_uint (global uint4 *reconstructed_buffer,
                global float *output,
                const float min,
                const float max)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);
    int output_offset = idz*4;

    float scale = (max-min)/255.0f;

    output[idx + idy*sizey + (output_offset)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].x)*scale+min;
    output[idx + idy*sizey + (output_offset+1)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].y)*scale+min;
    output[idx + idy*sizey + (output_offset+2)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].z)*scale+min;
    output[idx + idy*sizey + (output_offset+3)*sizex*sizey] = (reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey].w)*scale+min;
}

kernel void
backproject_tex (read_only image2d_t sinogram,
                 global float *slice,
                 constant float *sin_lut,
                 constant float *cos_lut,
                 const unsigned int x_offset,
                 const unsigned int y_offset,
                 const unsigned int angle_offset,
                 const unsigned int n_projections,
                 const float axis_pos)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    // const float bx = idx - axis_pos + x_offset;
    // const float by = idy - axis_pos + y_offset;
    const float bx = idx - axis_pos + x_offset + 0.5f;
    const float by = idy - axis_pos + y_offset + 0.5f;
    float sum = 0.0f;

#ifdef DEVICE_TESLA_K20XM
#pragma unroll 4
#endif
#ifdef DEVICE_TESLA_P100_PCIE_16GB
#pragma unroll 2
#endif
#ifdef DEVICE_GEFORCE_GTX_TITAN_BLACK
#pragma unroll 8
#endif
#ifdef DEVICE_GEFORCE_GTX_TITAN
#pragma unroll 14
#endif
#ifdef DEVICE_GEFORCE_GTX_1080_TI
#pragma unroll 10
#endif
#ifdef DEVICE_QUADRO_M6000
#pragma unroll 2
#endif
#ifdef DEVICE_GFX1010
#pragma unroll 4
#endif
    for(int proj = 0; proj < n_projections; proj++) {
        //float h = bx * cos_lut[angle_offset + proj] - by * sin_lut[angle_offset + proj] + axis_pos + 0.5f;
        float h = by * sin_lut[angle_offset + proj] + bx * cos_lut[angle_offset + proj] + axis_pos;
        sum += read_imagef (sinogram, volumeSampler, (float2)(h, proj + 0.5f)).x;
    }

    slice[idy * get_global_size(0) + idx] = sum * M_PI_F / n_projections;
}

kernel void
backproject_tex2d (
        read_only image2d_array_t sinogram,
        global float *slice,
        constant float *sin_lut,
        constant float *cos_lut,
        const unsigned int x_offset,
        const unsigned int y_offset,
        const unsigned int angle_offset,
        const unsigned int n_projections,
        const float axis_pos,
        unsigned long offset)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    const float bx = idx - axis_pos + x_offset + 0.5f;
    const float by = idy - axis_pos + y_offset + 0.5f;
    int output_offset = idz + (int)offset;
    float sum = 0.0f;
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

#ifdef DEVICE_TESLA_K20XM
#pragma unroll 4
#endif
#ifdef DEVICE_TESLA_P100_PCIE_16GB
#pragma unroll 2
#endif
#ifdef DEVICE_GEFORCE_GTX_TITAN_BLACK
#pragma unroll 8
#endif
#ifdef DEVICE_GEFORCE_GTX_TITAN
#pragma unroll 14
#endif
#ifdef DEVICE_GEFORCE_GTX_1080_TI
#pragma unroll 10
#endif
#ifdef DEVICE_QUADRO_M6000
#pragma unroll 2
#endif
#ifdef DEVICE_GFX1010
#pragma unroll 4
#endif
    for(int proj = 0; proj < n_projections; proj++) {
        float h = by * sin_lut[angle_offset + proj] + bx * cos_lut[angle_offset + proj] + axis_pos;
        sum += read_imagef (sinogram, volumeSampler, (float4)(h, proj + 0.5f, idz, 0.0)).x;
    }
    slice[idx + idy*sizey + output_offset*sizex*sizey] = sum * M_PI_F / n_projections;
}

kernel void
backproject_tex3d (
        read_only image2d_array_t sinogram,
        global float4 *reconstructed_buffer,
        constant float *sin_lut,
        constant float *cos_lut,
        const unsigned int x_offset,
        const unsigned int y_offset,
        const unsigned int angle_offset,
        const unsigned int n_projections,
        const float axis_pos){

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);

    const float bx = idx - axis_pos + x_offset + 0.5f;
    const float by = idy - axis_pos + y_offset + 0.5f;
    float4 sum = {0.0f,0.0f,0.0f,0.0f};

    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    for(int proj = 0; proj < n_projections; proj++) {
        float h = by * sin_lut[angle_offset + proj] + bx * cos_lut[angle_offset + proj] + axis_pos;
        sum += read_imagef (sinogram, volumeSampler, (float4)(h, proj + 0.5f,idz, 0.0));
    }
    reconstructed_buffer[idx + idy*sizey + idz*sizex*sizey] = sum * M_PI_F / n_projections;
}

kernel void
texture_float4 (
        read_only image2d_array_t sinogram,
        global float4 *reconstructed_buffer,
        constant float *sin_lut,
        constant float *cos_lut,
        const unsigned int x_offset,
        const unsigned int y_offset,
        const unsigned int angle_offset,
        const unsigned int n_projections,
        const float axis_pos){

        const int local_idx = get_local_id(0);
        const int local_idy = get_local_id(1);

        const int global_idx = get_global_id(0);
        const int global_idy = get_global_id(1);
        const int idz = get_global_id(2);

        int local_sizex = get_local_size(0);
        int local_sizey = get_local_size(1);

        int global_sizex = get_global_size(0);
        int global_sizey = get_global_size(1);

        /* Computing sequential numbers of 4x4 square, quadrant, and pixel within quadrant */
        int square = local_idy%4;
        int quadrant = local_idx/4;
        int pixel = local_idx%4;

        /* Computing projection and pixel offsets */
        int projection_index = local_idy/4;
        int2 remapped_index_local   = {(4*square + 2*(quadrant%2) + (pixel%2)),(2* (quadrant/2) + (pixel/2))};
        int2 remapped_index_global  = {(get_group_id(0)*get_local_size(0)+remapped_index_local.x),
                                        (get_group_id(1)*get_local_size(1)+remapped_index_local.y)};

        float2 pixel_coord = {(remapped_index_global.x-axis_pos), (remapped_index_global.y-axis_pos)}; //bx and by

        float4 sum[4] = {0.0f,0.0f,0.0f,0.0f};
        __local float4 shared_mem[64][4];
        __local float4 reconstructed_cache[16][16];


        for(int proj = projection_index; proj < n_projections; proj+=4) {
            float sine_value = sin_lut[angle_offset + proj];
            float h = pixel_coord.x * cos_lut[angle_offset + proj] - pixel_coord.y * sin_lut[angle_offset + proj] + axis_pos + 0.5f;
            for(int q=0; q<4; q+=1){
                   sum[q] += read_imagef(sinogram, volumeSampler, (float4)(h-4*q*sine_value, proj + 0.5f,idz, 0.0));
            }
        }

        int2 remapped_index = {(local_idx%4), (4*local_idy + (local_idx/4))};

        for(int q=0; q<4;q+=1){
            /* Moving partial sums to shared memory */
            shared_mem[(local_sizex*remapped_index_local.y + remapped_index_local.x)][projection_index] = sum[q];

            barrier(CLK_LOCAL_MEM_FENCE); // syncthreads

            for(int i=2; i>=1; i/=2){
                if(remapped_index.x <i){
                    shared_mem[remapped_index.y][remapped_index.x] += shared_mem[remapped_index.y][remapped_index.x+i];
                }
                barrier(CLK_GLOBAL_MEM_FENCE); // syncthreads
            }

            if(remapped_index.x == 0){
                reconstructed_cache[4*q+remapped_index.y/16][remapped_index.y%16] = shared_mem[remapped_index.y][0];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // syncthreads
        }

        reconstructed_buffer[global_idx + global_idy*global_sizey + idz*global_sizex*global_sizey] = reconstructed_cache[local_idy][local_idx];
}


kernel void
texture_float2 (
        read_only image2d_array_t sinogram,
        global float2 *reconstructed_buffer,
        constant float *sin_lut,
        constant float *cos_lut,
        const unsigned int x_offset,
        const unsigned int y_offset,
        const unsigned int angle_offset,
        const unsigned int n_projections,
        const float axis_pos){

    const int local_idx = get_local_id(0);
    const int local_idy = get_local_id(1);

    const int global_idx = get_global_id(0);
    const int global_idy = get_global_id(1);
    const int idz = get_global_id(2);

    int local_sizex = get_local_size(0);
    int local_sizey = get_local_size(1);

    int global_sizex = get_global_size(0);
    int global_sizey = get_global_size(1);

    /* Computing sequential numbers of 4x4 square, quadrant, and pixel within quadrant */
    int square = local_idy%4;
    int quadrant = local_idx/4;
    int pixel = local_idx%4;

    /* Computing projection and pixel offsets */
    int projection_index = local_idy/4;
    int2 remapped_index_local   = {(4*square + 2*(quadrant%2) + (pixel%2)),(2* (quadrant/2) + (pixel/2))};
    int2 remapped_index_global  = {(get_group_id(0)*get_local_size(0)+remapped_index_local.x),
                                    (get_group_id(1)*get_local_size(1)+remapped_index_local.y)};

    float2 pixel_coord = {(remapped_index_global.x-axis_pos), (remapped_index_global.y-axis_pos)};

    float2 sum[4] = {0.0f,0.0f};
    __local float2 shared_mem[64][4];
    __local float2 reconstructed_cache[16][16];


    for(int proj = projection_index; proj < n_projections; proj+=4) {
        float sine_value = sin_lut[angle_offset + proj];
        float h = axis_pos + pixel_coord.x * cos_lut[angle_offset + proj] - pixel_coord.y * sin_lut[angle_offset + proj] + 0.5f;
        for(int q=0; q<4; q+=1){
           sum[q] += read_imagef(sinogram, volumeSampler, (float4)(h-4*q*sine_value, proj + 0.5f,idz, 0.0)).xy;
        }
    }


     int2 remapped_index = {(local_idx%4), (4*local_idy + (local_idx/4))};

    for(int q=0; q<4;q+=1){
        /* Moving partial sums to shared memory */
        shared_mem[(local_sizex*remapped_index_local.y + remapped_index_local.x)][projection_index] = sum[q];

        barrier(CLK_LOCAL_MEM_FENCE); // syncthreads

        for(int i=2; i>=1; i/=2){
            if(remapped_index.x <i){
                shared_mem[remapped_index.y][remapped_index.x] += shared_mem[remapped_index.y][remapped_index.x+i];
            }
            barrier(CLK_GLOBAL_MEM_FENCE); // syncthreads
        }

        if(remapped_index.x == 0){
            reconstructed_cache[4*q+remapped_index.y/16][remapped_index.y%16] = shared_mem[remapped_index.y][0];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // syncthreads
    }

    reconstructed_buffer[global_idx + global_idy*global_sizey + idz*global_sizex*global_sizey] = reconstructed_cache[local_idy][local_idx];
}


kernel void
texture_uint (
        read_only image2d_array_t sinogram,
        global uint4 *reconstructed_buffer,
        constant float *sin_lut,
        constant float *cos_lut,
        const unsigned int x_offset,
        const unsigned int y_offset,
        const unsigned int angle_offset,
        const unsigned int n_projections,
        const float axis_pos){

        const int local_idx = get_local_id(0);
        const int local_idy = get_local_id(1);

        const int global_idx = get_global_id(0);
        const int global_idy = get_global_id(1);
        const int idz = get_global_id(2);

        int local_sizex = get_local_size(0);
        int local_sizey = get_local_size(1);

        int global_sizex = get_global_size(0);
        int global_sizey = get_global_size(1);

        /* Computing sequential numbers of 4x4 square, quadrant, and pixel within quadrant */
        int square = local_idy%4;
        int quadrant = local_idx/4;
        int pixel = local_idx%4;

        /* Computing projection and pixel offsets */
        int projection_index = local_idy/4;
        int2 remapped_index_local   = {(4*square + 2*(quadrant%2) + (pixel%2)),(2* (quadrant/2) + (pixel/2))};
        int2 remapped_index_global  = {(get_group_id(0)*get_local_size(0)+remapped_index_local.x),
                                        (get_group_id(1)*get_local_size(1)+remapped_index_local.y)};

        float2 pixel_coord = {(remapped_index_global.x-axis_pos), (remapped_index_global.y-axis_pos)}; //bx and by

        uint4 sum[4] = {0.0f,0.0f,0.0f,0.0f};
        __local uint4 shared_mem[64][4];
        __local uint4 reconstructed_cache[16][16];

        // uint4 read_imageui (	image2d_array_t  image,sampler_t  sampler,float4  coord )

        for(int proj = projection_index; proj < n_projections; proj+=4) {
            float sine_value = sin_lut[angle_offset + proj];
            float h = pixel_coord.x * cos_lut[angle_offset + proj] - pixel_coord.y * sin_lut[angle_offset + proj] + axis_pos + 0.5f;
            for(int q=0; q<4; q+=1){
               sum[q] += read_imageui(sinogram, volumeSampler, (float4)(h-4*q*sine_value, proj + 0.5f,idz, 0.0));
            }
        }


        int2 remapped_index = {(local_idx%4), (4*local_idy + (local_idx/4))};

        for(int q=0; q<4;q+=1){
            /* Moving partial sums to shared memory */
            shared_mem[(local_sizex*remapped_index_local.y + remapped_index_local.x)][projection_index] = sum[q];

            barrier(CLK_LOCAL_MEM_FENCE); // syncthreads

            for(int i=2; i>=1; i/=2){
                if(remapped_index.x <i){
                    shared_mem[remapped_index.y][remapped_index.x] += shared_mem[remapped_index.y][remapped_index.x+i];
                }
                barrier(CLK_GLOBAL_MEM_FENCE); // syncthreads
            }

            if(remapped_index.x == 0){
                reconstructed_cache[4*q+remapped_index.y/16][remapped_index.y%16] = shared_mem[remapped_index.y][0];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // syncthreads
        }

        reconstructed_buffer[global_idx + global_idy*global_sizey + idz*global_sizex*global_sizey] = reconstructed_cache[local_idy][local_idx];
}


kernel void sort(global float *input){
    int local_idx = get_local_id(0);
    int local_idy = get_local_id(1);

    uint group_sizex = get_local_size(0);
    uint group_sizey = get_local_size(1);

    int global_idx = get_global_id(0);
    int global_idy = get_global_id(1);

    __local float localMin[256];
    __local float localMax[256];
    __local float partialMin[4096];
    __local float partialMax[4096];

    localMax[local_idx + local_idy*group_sizex] = localMin[local_idx + local_idy*group_sizex] = input[global_idy * get_global_size(0) + global_idx];

    for(uint stride = (group_sizex*group_sizey)/2; stride > 0; stride /=2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if((local_idx + local_idy*group_sizex) < stride)
           localMin[local_idx + local_idy*group_sizex] = (localMin[local_idx + local_idy*group_sizex]<localMin[(local_idx + local_idy*group_sizex) + stride])?localMin[local_idx + local_idy*group_sizex]:localMin[local_idx + local_idy*group_sizex + stride];
           localMax[local_idx + local_idy*group_sizex] = (localMax[local_idx + local_idy*group_sizex]<localMax[(local_idx + local_idy*group_sizex) + stride])?localMax[local_idx + local_idy*group_sizex]:localMax[local_idx + local_idy*group_sizex + stride];
    }

    if((local_idx + local_idy*group_sizex)==0)
        partialMin[get_group_id(0)] = localMin[0];
        partialMax[get_group_id(0)] = localMax[0];
}

kernel void normalize_vec(global float *input_vec,
                      global unsigned int *normalized_vec,
                      const float min,
                      const float max){

    const float scale = 255.0f / (max - min);
    int index = get_global_id(0) + get_global_id(1) * get_global_size(1) + get_global_id(2) * get_global_size(0) * get_global_size(1);
    normalized_vec[index] = (unsigned int)((input_vec[index]-min)*scale);
}
