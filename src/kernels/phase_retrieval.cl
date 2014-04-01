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

#define METHOD_ROUTINE const int width = get_global_size(0);const int height = get_global_size(1);int idx = get_global_id(0);int idy = get_global_id(1);if (idx >= width || idy >= height) {return;}if (idx == 0 && idy == 0) {output[0] = 0.5f * pow(10, regularize_rate);return;}float n_idx = (idx > floor((float)width / 2.0f)) ? idx - width : idx;float n_idy = (idy > floor((float)height / 2.0f)) ? idy - height : idy;n_idx = n_idx / (1 - normalize * (1 - width));n_idy = n_idy / (1 - normalize * (1 - height));float sin_arg = prefac * (n_idy * n_idy + n_idx * n_idx) / 2.0f;float sin_value = sin(sin_arg);

__kernel void tie_method(int normalize, float prefac, float regularize_rate, float binary_filter_rate, __global float *output)
{
    METHOD_ROUTINE;
	output[idy * width + idx] = 0.5f / (sin_arg + pow(10, -regularize_rate));
}

__kernel void ctf_method(int normalize, float prefac, float regularize_rate, float binary_filter_rate, __global float *output) 
{
    METHOD_ROUTINE;
	output[idy * width + idx] = 0.5f * sign(sin_value) / (fabs(sin_value) + pow(10, -regularize_rate));
}

__kernel void ctfhalfsine_method(int normalize, float prefac, float regularize_rate, float binary_filter_rate, __global float *output) 
{
    METHOD_ROUTINE;
    output[idy * width + idx] = (sin_arg >= M_PI) ? 0.0f : 0.5f * sign(sin_value) / (fabs(sin_value) + 
														   pow(10, -regularize_rate));
}

__kernel void qp_method(int normalize, float prefac, float regularize_rate, float binary_filter_rate, __global float *output) 
{
    METHOD_ROUTINE;
    output[idy * width + idx] = (sin_arg > M_PI/2.0f && fabs(sin_value) < binary_filter_rate) ? 
								0.0f : 0.5f * sign(sin_value) / (fabs(sin_value) + pow(10, -regularize_rate));
}

__kernel void qphalfsine_method(int normalize, float prefac, float regularize_rate, float binary_filter_rate, __global float *output) 
{
    METHOD_ROUTINE;
    output[idy * width + idx] = ((sin_arg > M_PI/2.0f && fabs(sin_value) < binary_filter_rate) || sin_arg >= M_PI) ? 
								0.0f : 0.5f * sign(sin_value) / (fabs(sin_value) + pow(10, -regularize_rate));
}

__kernel void qp2_method(int normalize, float prefac, float regularize_rate, float binary_filter_rate, __global float *output) 
{
    METHOD_ROUTINE;
    float cacl_filter_value = 0.5f * sign(sin_value) / (fabs(sin_value) + pow(10, -regularize_rate));
    output[idy * width + idx] = (sin_arg > M_PI/2.0f && fabs(sin_value) < binary_filter_rate) ? 
								 sign(cacl_filter_value) / (2 * (binary_filter_rate + pow(10, -regularize_rate))) : cacl_filter_value;
}

__kernel void mult_by_value(__global float *output, __global float *values)
{
    int idx_o = get_global_id(1) * (2 * get_global_size(0)) + (2 * get_global_id(0));
    int idx_v = get_global_id(1) * get_global_size(0) + get_global_id(0);
    output[idx_o] = output[idx_o] * values[idx_v];
    output[idx_o + 1] = output[idx_o + 1] * values[idx_v];
}

__kernel void subtract_value(__global float *input, __global float *output, float value)
{
    int idx_o = get_global_id(1) * (2 * get_global_size(0)) + (2 * get_global_id(0));
    int idx_i = get_global_id(1) * get_global_size(0) + get_global_id(0);
    output[idx_o] = input[idx_i] - value;
    output[idx_o + 1] = 0.0f;
}

__kernel void get_real(__global float *input, __global float *output)
{
    int idx_i = get_global_id(1) * (2 * get_global_size(0)) + (2 * get_global_id(0));
    int idx_o = get_global_id(1) * get_global_size(0) + get_global_id(0);
    output[idx_o] = input[idx_i];
}
