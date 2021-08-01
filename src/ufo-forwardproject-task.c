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
#include "config.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <stdio.h>

#endif

#include "ufo-forwardproject-task.h"


struct _UfoForwardprojectTaskPrivate {
    cl_context context;
    cl_kernel kernel;
    cl_mem slice_mem;
    gfloat axis_pos;
    gfloat angle_step;
    guint num_projections;
    cl_kernel interleave_float4;
    cl_kernel texture_float4;
    cl_kernel uninterleave_float4;
    size_t out_mem_size;
};

static void ufo_task_interface_init (UfoTaskIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoForwardprojectTask, ufo_forwardproject_task, UFO_TYPE_TASK_NODE,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_TASK,
                                                ufo_task_interface_init))

#define UFO_FORWARDPROJECT_TASK_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_FORWARDPROJECT_TASK, UfoForwardprojectTaskPrivate))

enum {
    PROP_0,
    PROP_AXIS_POSITION,
    PROP_ANGLE_STEP,
    PROP_NUM_PROJECTIONS,
    N_PROPERTIES
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

UfoNode *
ufo_forwardproject_task_new (void)
{
    return UFO_NODE (g_object_new (UFO_TYPE_FORWARDPROJECT_TASK, NULL));
}

static void
ufo_forwardproject_task_setup (UfoTask *task,
                               UfoResources *resources,
                               GError **error)
{
    UfoForwardprojectTaskPrivate *priv;

    priv = UFO_FORWARDPROJECT_TASK (task)->priv;
    priv->context = ufo_resources_get_context(resources);
    priv->kernel = ufo_resources_get_kernel (resources, "forwardproject.cl", "forwardproject", NULL, error);

    priv->interleave_float4 = ufo_resources_get_kernel (resources, "forwardproject.cl", "interleave_float4", NULL, error);
    priv->texture_float4 = ufo_resources_get_kernel (resources, "forwardproject.cl", "texture_float4", NULL, error);
    priv->uninterleave_float4 = ufo_resources_get_kernel (resources, "forwardproject.cl", "uninterleave_float4", NULL, error);

    if (priv->kernel != NULL)
        UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->kernel), error);

    if (priv->interleave_float4 != NULL)
        UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->interleave_float4), error);

    if (priv->texture_float4 != NULL)
        UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->texture_float4), error);

    if (priv->uninterleave_float4 != NULL)
        UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->uninterleave_float4), error);

    if (priv->angle_step == 0) 
        priv->angle_step = G_PI / priv->num_projections;
}

static void
ufo_forwardproject_task_get_requisition (UfoTask *task,
                                         UfoBuffer **inputs,
                                         UfoRequisition *requisition,
                                         GError **error)
{
    UfoForwardprojectTaskPrivate *priv;
    UfoRequisition in_req;

    priv = UFO_FORWARDPROJECT_TASK (task)->priv;

    ufo_buffer_get_requisition (inputs[0], &in_req);

    requisition->n_dims  = in_req.n_dims;
    requisition->dims[0] = in_req.dims[0];
    requisition->dims[1] = priv->num_projections;
    requisition->dims[2] = in_req.dims[2];

/*    requisition->n_dims = 2;
    requisition->dims[0] = in_req.dims[0];
    requisition->dims[1] = priv->num_projections;*/
    if (priv->axis_pos == -G_MAXFLOAT) {
        priv->axis_pos = in_req.dims[0] / 2.0f;
    }
}

static guint
ufo_forwardproject_task_get_num_inputs (UfoTask *task)
{
    return 1;
}

static guint
ufo_forwardproject_task_get_num_dimensions (UfoTask *task,
                               guint input)
{
    g_return_val_if_fail (input == 0, 0);
//    return 2;
    return 3;
}

static UfoTaskMode
ufo_forwardproject_task_get_mode (UfoTask *task)
{
    return UFO_TASK_MODE_PROCESSOR | UFO_TASK_MODE_GPU;
}

static gboolean
ufo_forwardproject_task_process (UfoTask *task,
                                 UfoBuffer **inputs,
                                 UfoBuffer *output,
                                 UfoRequisition *requisition)
{
    UfoForwardprojectTaskPrivate *priv;
    UfoGpuNode *node;
    UfoProfiler *profiler;
    cl_command_queue cmd_queue;
    cl_mem in_mem;
    cl_mem out_mem;
    cl_mem device_array;
    cl_mem interleaved_img;
    cl_kernel kernel_interleave;
    cl_kernel kernel_texture;
    cl_kernel kernel_uninterleave;
    cl_mem reconstructed_buffer;

    priv = UFO_FORWARDPROJECT_TASK (task)->priv;
    node = UFO_GPU_NODE (ufo_task_node_get_proc_node (UFO_TASK_NODE (task)));
    cmd_queue = ufo_gpu_node_get_cmd_queue (node);
    out_mem = ufo_buffer_get_device_array (output, cmd_queue);
    profiler = ufo_task_node_get_profiler (UFO_TASK_NODE (task));

    ufo_profiler_enable_tracing(profiler,TRUE);

/*    fprintf(stdout, "N_Dimensions: %u \n",requisition->n_dims);
    fprintf(stdout, "Dim-0: %lu \t Dim-1: %lu \t Dim-2: %lu \n",requisition->dims[0],requisition->dims[1],requisition->dims[2]);
    fprintf(stdout, "Axis pos: %f \n",priv->axis_pos);
    fprintf(stdout, "Angle step: %f \n",priv->angle_step);*/

    if(requisition->n_dims == 2) {
        in_mem = ufo_buffer_get_device_image (inputs[0], cmd_queue);

        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(priv->kernel, 0, sizeof(cl_mem), &in_mem));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(priv->kernel, 1, sizeof(cl_mem), &out_mem));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(priv->kernel, 2, sizeof(gfloat), &priv->axis_pos));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(priv->kernel, 3, sizeof(gfloat), &priv->angle_step));

        ufo_profiler_call(profiler, cmd_queue, priv->kernel, 2, requisition->dims, NULL);

        gfloat *hostData;
        hostData = (gfloat*) malloc(sizeof(float)*requisition->dims[0]*requisition->dims[1]);

        clEnqueueReadBuffer(cmd_queue,out_mem,CL_TRUE,0,sizeof(float)*requisition->dims[0]*requisition->dims[1],
                             hostData,0,NULL,NULL);

        float sum = 0.0f;
        for(size_t i=0; i<requisition->dims[0]*requisition->dims[1]; i++){
            sum += hostData[i];
        }
        fprintf(stdout, "Sum: %f \n",sum);

    } else{

        // Quotient
        unsigned long quotient;
        quotient = requisition->dims[2]/4;

        // Image format
        cl_image_format format;
        format.image_channel_data_type = CL_FLOAT;
        format.image_channel_order = CL_RGBA;

        // Image Description
        cl_image_desc imageDesc;
        imageDesc.image_width = requisition->dims[0];
        imageDesc.image_height = requisition->dims[1];
        imageDesc.image_depth = 0;
        imageDesc.image_array_size = quotient;
        imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        imageDesc.image_slice_pitch = 0;
        imageDesc.image_row_pitch = 0;
        imageDesc.num_mip_levels = 0;
        imageDesc.num_samples = 0;
        imageDesc.buffer = NULL;

        // Interleave
        device_array = ufo_buffer_get_device_array (inputs[0], cmd_queue);
        interleaved_img = clCreateImage(priv->context,CL_MEM_READ_WRITE,&format,&imageDesc,NULL,0);

        kernel_interleave = priv->interleave_float4;
        clSetKernelArg(kernel_interleave, 0, sizeof(cl_mem), &device_array);
        clSetKernelArg(kernel_interleave, 1, sizeof(cl_mem), &interleaved_img);

        size_t gWorkSize_3d[3] = {requisition->dims[0],requisition->dims[1],quotient};
        ufo_profiler_call(profiler, cmd_queue, kernel_interleave, 3, gWorkSize_3d, NULL);
//        cl_int err = clEnqueueNDRangeKernel(cmd_queue,kernel_interleave,3,0,gWorkSize_3d,NULL,0,NULL,NULL);
//        fprintf(stdout, "Error Interleave: %d \n",err);

        // Forward projection
        size_t buffer_size = sizeof(cl_float4) * requisition->dims[0] * requisition->dims[1] * quotient;
        reconstructed_buffer = clCreateBuffer(priv->context, CL_MEM_READ_WRITE, buffer_size, NULL, 0);

        kernel_texture = priv->texture_float4;
        clSetKernelArg(kernel_texture, 0, sizeof(cl_mem), &interleaved_img);
        clSetKernelArg(kernel_texture, 1, sizeof(cl_mem), &reconstructed_buffer);
        clSetKernelArg(kernel_texture, 2, sizeof(gfloat), &priv->axis_pos);
        clSetKernelArg(kernel_texture, 3, sizeof(gfloat), &priv->angle_step);

        size_t gSize[3] = {requisition->dims[0], requisition->dims[1], quotient};
        size_t lSize[3] = {16, 16, 1};
        ufo_profiler_call(profiler, cmd_queue, kernel_texture, 3, gSize, lSize);
//        err = clEnqueueNDRangeKernel(cmd_queue,kernel_texture,3,0,gSize,lSize,0,NULL,NULL);
//        fprintf(stdout, "Error Texture: %d \n",err);

        // Uninterleave
        kernel_uninterleave = priv->uninterleave_float4;
        clSetKernelArg(kernel_uninterleave, 0, sizeof(cl_mem), &reconstructed_buffer);
        clSetKernelArg(kernel_uninterleave, 1, sizeof(cl_mem), &out_mem);
        ufo_profiler_call(profiler, cmd_queue, kernel_uninterleave, 3, gWorkSize_3d, NULL);
//        err = clEnqueueNDRangeKernel(cmd_queue,kernel_uninterleave,3,0,gWorkSize_3d,NULL,0,NULL,NULL);
//        fprintf(stdout, "Error Uninterleave: %d \n",err);

        gfloat *hostData;
        hostData = (gfloat*) malloc(sizeof(float)*requisition->dims[0]*requisition->dims[1]*requisition->dims[2]);

        clEnqueueReadBuffer(cmd_queue,out_mem,CL_TRUE,0,sizeof(float)*requisition->dims[0]*requisition->dims[1]*requisition->dims[2],
                             hostData,0,NULL,NULL);

        float sum = 0.0f;
        for(size_t i=0; i<requisition->dims[0]*requisition->dims[1]*requisition->dims[2]; i++){
            sum += hostData[i];
        }
        fprintf(stdout, "Sum: %f \n",sum);

        clReleaseMemObject(interleaved_img);
        clReleaseMemObject(reconstructed_buffer);
    }
    size_t temp_size;
    clGetMemObjectInfo(out_mem, CL_MEM_SIZE,
                       sizeof(temp_size), &temp_size, NULL);
    priv->out_mem_size += temp_size;
    fprintf(stdout, "Time taken GPU: %f Size: %zu \n", ufo_profiler_elapsed(profiler,UFO_PROFILER_TIMER_GPU),priv->out_mem_size);


    return TRUE;
}

static void
ufo_forwardproject_task_set_property (GObject *object,
                                      guint property_id,
                                      const GValue *value,
                                      GParamSpec *pspec)
{
    UfoForwardprojectTaskPrivate *priv = UFO_FORWARDPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_AXIS_POSITION:
            priv->axis_pos = g_value_get_float (value);
            break;
        case PROP_ANGLE_STEP:
            priv->angle_step = g_value_get_float(value);
            break;
        case PROP_NUM_PROJECTIONS:
            priv->num_projections = g_value_get_uint(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_forwardproject_task_get_property (GObject *object,
                                      guint property_id,
                                      GValue *value,
                                      GParamSpec *pspec)
{
    UfoForwardprojectTaskPrivate *priv = UFO_FORWARDPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_AXIS_POSITION:
            g_value_set_float (value, priv->axis_pos);
            break;
        case PROP_ANGLE_STEP:
            g_value_set_float(value, priv->angle_step);
            break;
        case PROP_NUM_PROJECTIONS:
            g_value_set_uint(value, priv->num_projections);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_forwardproject_task_finalize (GObject *object)
{
    UfoForwardprojectTaskPrivate *priv;

    priv = UFO_FORWARDPROJECT_TASK_GET_PRIVATE (object);

    if (priv->kernel) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->kernel));
        priv->kernel = NULL;
    }

    if (priv->interleave_float4) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->interleave_float4));
        priv->interleave_float4 = NULL;
    }

    if (priv->texture_float4) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->texture_float4));
        priv->texture_float4 = NULL;
    }

    if (priv->uninterleave_float4) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->uninterleave_float4));
        priv->uninterleave_float4 = NULL;
    }

    G_OBJECT_CLASS (ufo_forwardproject_task_parent_class)->finalize (object);
}

static void
ufo_task_interface_init (UfoTaskIface *iface)
{
    iface->setup = ufo_forwardproject_task_setup;
    iface->get_requisition = ufo_forwardproject_task_get_requisition;
    iface->get_num_inputs = ufo_forwardproject_task_get_num_inputs;
    iface->get_num_dimensions = ufo_forwardproject_task_get_num_dimensions;
    iface->get_mode = ufo_forwardproject_task_get_mode;
    iface->process = ufo_forwardproject_task_process;
}

static void
ufo_forwardproject_task_class_init (UfoForwardprojectTaskClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);

    gobject_class->set_property = ufo_forwardproject_task_set_property;
    gobject_class->get_property = ufo_forwardproject_task_get_property;
    gobject_class->finalize = ufo_forwardproject_task_finalize;

    properties[PROP_AXIS_POSITION] =
        g_param_spec_float ("axis-pos",
            "Position of rotation axis",
            "Position of rotation axis",
            -G_MAXFLOAT, G_MAXFLOAT, -G_MAXFLOAT,
            G_PARAM_READWRITE);

    properties[PROP_ANGLE_STEP] =
        g_param_spec_float("angle-step",
            "Increment of angle in radians",
            "Increment of angle in radians",
            -4.0f * ((gfloat) G_PI),
            +4.0f * ((gfloat) G_PI),
            0.0f,
            G_PARAM_READWRITE);

    properties[PROP_NUM_PROJECTIONS] =
        g_param_spec_uint("number",
            "Number of projections",
            "Number of projections",
            1, 32768, 256,
            G_PARAM_READWRITE);

    for (guint i = PROP_0 + 1; i < N_PROPERTIES; i++)
        g_object_class_install_property (gobject_class, i, properties[i]);

    g_type_class_add_private (gobject_class, sizeof(UfoForwardprojectTaskPrivate));
}

static void
ufo_forwardproject_task_init(UfoForwardprojectTask *self)
{
    self->priv = UFO_FORWARDPROJECT_TASK_GET_PRIVATE(self);
    self->priv->kernel = NULL;
    self->priv->interleave_float4 = NULL;
    self->priv->uninterleave_float4 = NULL;
    self->priv->texture_float4 = NULL;
    self->priv->axis_pos = -G_MAXFLOAT;
    self->priv->num_projections = 256;
    self->priv->angle_step = 0;
    self->priv->out_mem_size = 0;
}
