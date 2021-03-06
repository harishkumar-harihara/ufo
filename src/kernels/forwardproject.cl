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

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE |
                             CLK_FILTER_NEAREST             ;

kernel void
forwardproject(read_only image2d_t slice,
               global float *sinogram,
               float axis_pos,
               float angle_step)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int slice_width = get_global_size(0);

    const float angle = idy * angle_step;
    /* radius of object circle */
    const float r = fmin (axis_pos, slice_width - axis_pos);
    /* positive/negative distance from detector center */
    const float d = idx - axis_pos + 0.5f;
    /* length of the cut through the circle */
    const float l = sqrt(4.0f*r*r - 4.0f*d*d);

    /* vector in detector direction */
    float2 D = (float2) (cos(angle), sin(angle));
    D = normalize(D);

    /* vector perpendicular to the detector */
    const float2 N = (float2) (D.y, -D.x);

    /* sample point in the circle traveling along N */
    float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));
    float sum = 0.0f;

    for (int i = 0; i < l; i++) {
        sum += read_imagef(slice, sampler, sample).x;
        sample += N;
    }

    sinogram[idy * slice_width + idx] = sum;
}

kernel void
interleave_single (global float *slice,
                   write_only image2d_array_t interleaved_slice)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    int slice_offset = idz*2;

    float x = slice[idx + idy * sizex + (slice_offset) * sizex * sizey];
    float y = slice[idx + idy * sizex + (slice_offset+1) * sizex * sizey];

    write_imagef(interleaved_slice, (int4)(idx, idy, idz, 0),(float4)(x,y,0.0f,0.0f));
}

kernel void
forwardproject_tex3d (
        read_only image2d_array_t slice,
        global float2 *reconstructed_sinogram,
        float axis_pos,
        float angle_step) {

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);

    const float angle = idy * angle_step;
    const float r = fmin (axis_pos, sizex - axis_pos);
    const float d = idx - axis_pos + 0.5f;
    const float l = sqrt(4.0f*r*r - 4.0f*d*d);

    float2 D = (float2) (cos(angle), sin(angle));
    D = normalize(D);

    const float2 N = (float2) (D.y, -D.x);

    float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));
    float2 sum = {0.0f,0.0f};

    for (int i = 0; i < l; i++) {
            sum += read_imagef(slice, sampler, (float4)((float2)sample,idz,0.0f)).xy;
            sample += N;
    }

    reconstructed_sinogram[idx + idy*sizex + idz*sizex*sizey] = sum;
}

kernel void
texture_single (
        read_only image2d_array_t slice,
        global float2 *reconstructed_sinogram,
        float axis_pos,
        float angle_step) {

    const int local_idx = get_local_id(0);
    const int local_idy = get_local_id(1);

    const int global_idx = get_global_id(0);
    const int global_idy = get_global_id(1);
    const int idz = get_global_id(2);

    int local_sizex = get_local_size(0);
    int local_sizey = get_local_size(1);

    int global_sizex = get_global_size(0);
    int global_sizey = get_global_size(1);

    int square = local_idy%4;
    int quadrant = local_idx/4;
    int pixel = local_idx%4;

    int projection_index = local_idy/4;
    int2 remapped_index_local   = {(4*square + 2*(quadrant%2) + (pixel%2)),(2* (quadrant/2) + (pixel/2))};
    int2 remapped_index_global  = {(get_group_id(0)*get_local_size(0)+remapped_index_local.x),
                                   (get_group_id(1)*get_local_size(1)+remapped_index_local.y)};


    const float angle = remapped_index_global.y * angle_step;
    const float r = fmin (axis_pos, global_sizex - axis_pos);
    const float d = remapped_index_global.x - axis_pos + 0.5f;
    const float l = sqrt(4.0f*r*r - 4.0f*d*d);

    float2 D = (float2) (cos(angle), sin(angle));
    D = normalize(D);

    const float2 N = (float2) (D.y, -D.x);
    float2 sample = d * D - l/2.0f * N + ((float2) (axis_pos, axis_pos));

    float2 sum[4] = {0.0f,0.0f};
    __local float2 shared_mem[64][4];
    __local float2 reconstructed_cache[16][16];

    for (int i = projection_index; i < l; i+=4) {
        for(int q=0; q<4; q+=1){
            sum[q] += read_imagef(slice, sampler, (float4)((float2)sample,idz,0.0f)).xy;
            sample += N; // must be in this loop, else only one half of sinogram is constructed
        }
    }

    int2 remapped_index = {(local_idx%4), (4*local_idy + (local_idx/4))};

    for(int q=0; q<4;q+=1){
        // Moving partial sums to shared memory
        shared_mem[(local_sizex*remapped_index_local.y + remapped_index_local.x)][projection_index] = sum[q];

        barrier(CLK_LOCAL_MEM_FENCE); // syncthreads

        for(int i=2; i>=1; i/=2){
            if(remapped_index.x < i){
                shared_mem[remapped_index.y][remapped_index.x] += shared_mem[remapped_index.y][remapped_index.x+i];
            }
            barrier(CLK_GLOBAL_MEM_FENCE); // syncthreads
        }

        if(remapped_index.x == 0){
            reconstructed_cache[4*q+remapped_index.y/16][remapped_index.y%16] = shared_mem[remapped_index.y][0];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // syncthreads
    }

    reconstructed_sinogram[global_idx + global_idy*global_sizex + idz*global_sizex*global_sizey] = reconstructed_cache[local_idy][local_idx];
}

kernel void
uninterleave_single (global float2 *reconstructed_sinogram,
                     global float *sinogram)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int idz = get_global_id(2);
    const int sizex = get_global_size(0);
    const int sizey = get_global_size(1);
    int output_offset = idz*2;

    sinogram[idx + idy*sizex + (output_offset)*sizex*sizey] = reconstructed_sinogram[idx + idy*sizex + idz*sizex*sizey].x;
    sinogram[idx + idy*sizex + (output_offset+1)*sizex*sizey] = reconstructed_sinogram[idx + idy*sizex + idz*sizex*sizey].y;
}


