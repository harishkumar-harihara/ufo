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
#endif

#include <math.h>
#include <stdio.h>
#include "ufo-backproject-task.h"


typedef enum {
    MODE_NEAREST,
    MODE_TEXTURE
} Mode;

static GEnumValue mode_values[] = {
        { MODE_NEAREST, "MODE_NEAREST", "nearest" },
        { MODE_TEXTURE, "MODE_TEXTURE", "texture" },
        { 0, NULL, NULL}
};

struct _UfoBackprojectTaskPrivate {
    cl_context context;
    cl_kernel nearest_kernel;
    cl_kernel texture_kernel;
    cl_kernel optimized_kernel_2d;
    cl_kernel optimized_kernel_3d;
    cl_kernel interleave;
    cl_kernel uninterleave;
    cl_mem sin_lut;
    cl_mem cos_lut;
    gfloat *host_sin_lut;
    gfloat *host_cos_lut;
    gdouble axis_pos;
    gdouble angle_step;
    gdouble angle_offset;
    gdouble real_angle_step;
    gboolean luts_changed;
    guint offset;
    guint burst_projections;
    guint n_projections;
    guint roi_x;
    guint roi_y;
    gint roi_width;
    gint roi_height;
    Mode mode;
};

static void ufo_task_interface_init (UfoTaskIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoBackprojectTask, ufo_backproject_task, UFO_TYPE_TASK_NODE,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_TASK,
                                                ufo_task_interface_init))

#define UFO_BACKPROJECT_TASK_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_BACKPROJECT_TASK, UfoBackprojectTaskPrivate))

enum {
    PROP_0,
    PROP_NUM_PROJECTIONS,
    PROP_OFFSET,
    PROP_AXIS_POSITION,
    PROP_ANGLE_STEP,
    PROP_ANGLE_OFFSET,
    PROP_ROI_X,
    PROP_ROI_Y,
    PROP_ROI_WIDTH,
    PROP_ROI_HEIGHT,
    PROP_MODE,
    N_PROPERTIES
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

UfoNode *
ufo_backproject_task_new (void)
{
    return UFO_NODE (g_object_new (UFO_TYPE_BACKPROJECT_TASK, NULL));
}

static cl_ulong
get_time_stamps (cl_event event)
{
    cl_ulong *start = 0, *end=0;
    UFO_RESOURCES_CHECK_CLERR (clWaitForEvents (1, &event));
    UFO_RESOURCES_CHECK_CLERR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_START, sizeof (cl_ulong), start, NULL));
    UFO_RESOURCES_CHECK_CLERR (clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_END, sizeof (cl_ulong), end, NULL));
    return end-start;
}

static gboolean
ufo_backproject_task_process (UfoTask *task,
                              UfoBuffer **inputs,
                              UfoBuffer *output,
                              UfoRequisition *requisition)
{
    UfoBackprojectTaskPrivate *priv;
    UfoGpuNode *node;
    UfoProfiler *profiler;
    cl_command_queue cmd_queue;
    cl_mem in_mem;
    cl_mem img_2d;
    cl_mem interleaved_img;
    cl_mem out_mem;
//    cl_mem interleaved_buffer;
    cl_mem reconstructed_buffer;
    cl_kernel kernel;
    gfloat axis_pos;

    priv = UFO_BACKPROJECT_TASK (task)->priv;
    node = UFO_GPU_NODE (ufo_task_node_get_proc_node(UFO_TASK_NODE(task)));
    cmd_queue = ufo_gpu_node_get_cmd_queue(node);
    out_mem = ufo_buffer_get_device_array (output, cmd_queue);
    profiler = ufo_task_node_get_profiler (UFO_TASK_NODE (task));

    ufo_profiler_enable_tracing(profiler,TRUE);

    /* Guess axis position if they are not provided by the user. */
    if (priv->axis_pos <= 0.0) {
        UfoRequisition in_req;

        ufo_buffer_get_requisition(inputs[0], &in_req);
        axis_pos = (gfloat) ((gfloat) in_req.dims[0]) / 2.0f;
    } else {
        axis_pos = priv->axis_pos;
    }

    if(requisition->n_dims == 2){
        //older variant
        if (priv->mode == MODE_TEXTURE) {
            in_mem = ufo_buffer_get_device_image (inputs[0], cmd_queue);
            kernel = priv->texture_kernel;
        }
        else {
            in_mem = ufo_buffer_get_device_array (inputs[0], cmd_queue);
            kernel = priv->nearest_kernel;
        }

        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &in_mem));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 1, sizeof (cl_mem), &out_mem));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 2, sizeof (cl_mem), &priv->sin_lut));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 3, sizeof (cl_mem), &priv->cos_lut));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 4, sizeof (guint),  &priv->roi_x));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 5, sizeof (guint),  &priv->roi_y));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 6, sizeof (guint),  &priv->offset));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 7, sizeof (guint),  &priv->burst_projections));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel, 8, sizeof (gfloat), &axis_pos));

        ufo_profiler_call (profiler, cmd_queue, kernel, 2, requisition->dims, NULL);
    }
    else{

        // Image format
        cl_image_format format;
        format.image_channel_order = CL_INTENSITY;
        format.image_channel_data_type = CL_FLOAT;

        // Region to copy
        size_t region[3];
        region[0] = requisition->dims[0];
        region[1] = requisition->dims[1];
        region[2] = 1;

        //Source and Destination origins
        size_t src_origin[3];
        src_origin[0] =  0;
        src_origin[1] =  0;

        size_t dst_origin[] = { 0, 0, 0 };

        cl_mem device_array = ufo_buffer_get_device_array(inputs[0],cmd_queue);

        // Setting global work size
        gsize gWorkSize_3d[3];
        gsize gWorkSize_2d[2];
        gWorkSize_2d[0] = gWorkSize_3d[0] = requisition->dims[0];
        gWorkSize_2d[1] = gWorkSize_3d[1] = requisition->dims[1];

        unsigned long quotient = requisition->dims[2]/4;
        unsigned long remainder = requisition->dims[2] % 4;
        unsigned long offset = requisition->dims[2] - remainder;

        size_t regionToCopy[3];
        regionToCopy[0] = requisition->dims[0];
        regionToCopy[1] = requisition->dims[1];

            /* PROCESS INTERLEAVE STAGE */
            if(quotient > 0) {
                kernel = priv->interleave;

                // output of interleave stage, each 4 slice is packed into a single slice with 4 elements
                // so 8 slices are packed into 2 Z slice with 4 elements
                interleaved_img = clCreateImage3D(priv->context, CL_MEM_READ_WRITE, &format,
                                                  requisition->dims[0], requisition->dims[1], offset,
                                                  0, 0, NULL, 0);

                clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_array);
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &interleaved_img);

                gWorkSize_3d[2] = requisition->dims[2];

                ufo_profiler_call(profiler, cmd_queue, kernel, 3, gWorkSize_3d, NULL);

                /* SINOGRAM RECONSTRUCTION FOR MULTIPLE SLICES */
                kernel = priv->optimized_kernel_3d;

                reconstructed_buffer = clCreateBuffer(priv->context, CL_MEM_READ_WRITE,
                                                      sizeof(cl_float4) * requisition->dims[0] * requisition->dims[1] *
                                                      quotient, NULL, 0);


                clSetKernelArg(kernel, 0, sizeof(cl_mem), &interleaved_img);
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &reconstructed_buffer);
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &priv->sin_lut);
                clSetKernelArg(kernel, 3, sizeof(cl_mem), &priv->cos_lut);
                clSetKernelArg(kernel, 4, sizeof(guint), &priv->roi_x);
                clSetKernelArg(kernel, 5, sizeof(guint), &priv->roi_y);
                clSetKernelArg(kernel, 6, sizeof(guint), &priv->offset);
                clSetKernelArg(kernel, 7, sizeof(guint), &priv->burst_projections);
                clSetKernelArg(kernel, 8, sizeof(gfloat), &axis_pos);

                size_t gSize[3] = {requisition->dims[0], requisition->dims[1], quotient};

                ufo_profiler_call(profiler, cmd_queue, kernel, 3, gSize, NULL);


                /*UNINTERLEAVE*/
                kernel = priv->uninterleave;

                clSetKernelArg(kernel, 0, sizeof(cl_mem), &reconstructed_buffer);
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem);

                size_t gSize_uninterleave[3] = {requisition->dims[0], requisition->dims[1], quotient};

                ufo_profiler_call(profiler, cmd_queue, kernel, 3, gSize_uninterleave, NULL);
                clReleaseMemObject(interleaved_img);
                clReleaseMemObject(reconstructed_buffer);
            }


            /* RECONSTRUCT 2D SLICES */
            if(remainder > 0) {
                kernel = priv->optimized_kernel_2d;

                img_2d = clCreateImage3D(priv->context, CL_MEM_READ_WRITE, &format,
                                         requisition->dims[0], requisition->dims[1], remainder,
                                         0, 0, NULL, 0);

                regionToCopy[2] = remainder;
                clEnqueueCopyBufferToImage(cmd_queue, device_array, img_2d,
                                           offset * requisition->dims[0] * requisition->dims[1],
                                           dst_origin, regionToCopy, 0, NULL, NULL);

                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_2d));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 2, sizeof(cl_mem), &priv->sin_lut));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 3, sizeof(cl_mem), &priv->cos_lut));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 4, sizeof(guint), &priv->roi_x));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 5, sizeof(guint), &priv->roi_y));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 6, sizeof(guint), &priv->offset));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 7, sizeof(guint), &priv->burst_projections));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 8, sizeof(gfloat), &axis_pos));
                UFO_RESOURCES_CHECK_CLERR (clSetKernelArg(kernel, 9, sizeof(unsigned long), &offset));

                gWorkSize_3d[2] = remainder;
                ufo_profiler_call(profiler, cmd_queue, kernel, 3, gWorkSize_3d, NULL);
                clReleaseMemObject(img_2d);
            }
    }

    fprintf(stdout, "Time taken GPU: %f \n", ufo_profiler_elapsed(profiler,UFO_PROFILER_TIMER_GPU));
    return TRUE;
}

static void
ufo_backproject_task_setup (UfoTask *task,
                            UfoResources *resources,
                            GError **error)
{
    UfoBackprojectTaskPrivate *priv;

    priv = UFO_BACKPROJECT_TASK_GET_PRIVATE (task);

    priv->context = ufo_resources_get_context (resources);
    priv->nearest_kernel = ufo_resources_get_kernel (resources, "backproject.cl", "backproject_nearest", NULL, error);
    priv->texture_kernel = ufo_resources_get_kernel (resources, "backproject.cl", "backproject_tex", NULL, error);
    priv->optimized_kernel_3d = ufo_resources_get_kernel (resources, "backproject.cl", "backproject_tex3d", NULL, error);
    priv->optimized_kernel_2d = ufo_resources_get_kernel (resources, "backproject.cl", "backproject_tex2d", NULL, error);
    priv->interleave = ufo_resources_get_kernel (resources, "backproject.cl", "interleave", NULL, error);
    priv->uninterleave = ufo_resources_get_kernel (resources, "backproject.cl", "uninterleave", NULL, error);

    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainContext (priv->context), error);

    if (priv->nearest_kernel != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->nearest_kernel), error);

    if (priv->texture_kernel != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->texture_kernel), error);

    if (priv->optimized_kernel_3d != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->optimized_kernel_3d), error);

    if (priv->optimized_kernel_2d != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->optimized_kernel_2d), error);

    if (priv->interleave != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->interleave), error);

    if (priv->uninterleave != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->uninterleave), error);
}

static cl_mem
create_lut_buffer (UfoBackprojectTaskPrivate *priv,
                   gfloat **host_mem,
                   gsize n_entries,
                   double (*func)(double))
{
    cl_int errcode;
    gsize size = n_entries * sizeof (gfloat);
    cl_mem mem = NULL;

    *host_mem = g_realloc (*host_mem, size);

    for (guint i = 0; i < n_entries; i++)
        (*host_mem)[i] = (gfloat) func (priv->angle_offset + i * priv->real_angle_step);

    mem = clCreateBuffer (priv->context,
                          CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                          size, *host_mem,
                          &errcode);

    UFO_RESOURCES_CHECK_CLERR (errcode);
    return mem;
}

static void
release_lut_mems (UfoBackprojectTaskPrivate *priv)
{
    if (priv->sin_lut) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseMemObject (priv->sin_lut));
        priv->sin_lut = NULL;
    }

    if (priv->cos_lut) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseMemObject (priv->cos_lut));
        priv->cos_lut = NULL;
    }
}

static void
ufo_backproject_task_get_requisition (UfoTask *task,
                                      UfoBuffer **inputs,
                                      UfoRequisition *requisition,
                                      GError **error)
{
    UfoBackprojectTaskPrivate *priv;
    UfoRequisition in_req;

    priv = UFO_BACKPROJECT_TASK_GET_PRIVATE (task);
    ufo_buffer_get_requisition (inputs[0], &in_req);

    /* If the number of projections is not specified use the input size */
    if (priv->n_projections == 0) {
        priv->n_projections = (guint) in_req.dims[1];
    }

    priv->burst_projections = (guint) in_req.dims[1];

    if (priv->burst_projections > priv->n_projections) {
        g_set_error (error, UFO_TASK_ERROR, UFO_TASK_ERROR_GET_REQUISITION,
                     "Total number of projections (%u) must be greater than "
                     "or equal to sinogram height (%u)",
                     priv->n_projections, priv->burst_projections);
        return;
    }

    requisition->n_dims = in_req.n_dims;

    /* TODO: we should check here, that we might access data outside the
     * projections */
    requisition->dims[0] = priv->roi_width == 0 ? in_req.dims[0] : (gsize) priv->roi_width;
    requisition->dims[1] = priv->roi_height == 0 ? in_req.dims[0] : (gsize) priv->roi_height;
    requisition->dims[2] = in_req.n_dims == 3 ? in_req.dims[2]:1;

    if (priv->real_angle_step < 0.0) {
        if (priv->angle_step <= 0.0)
            priv->real_angle_step = G_PI / ((gdouble) priv->n_projections);
        else
            priv->real_angle_step = priv->angle_step;
    }

    if (priv->luts_changed) {
        release_lut_mems (priv);
        priv->luts_changed = FALSE;
    }

    if (priv->sin_lut == NULL) {
        priv->sin_lut = create_lut_buffer (priv, &priv->host_sin_lut,
                                           priv->n_projections, sin);
    }

    if (priv->cos_lut == NULL) {
        priv->cos_lut = create_lut_buffer (priv, &priv->host_cos_lut,
                                           priv->n_projections, cos);
    }
}

static guint
ufo_filter_task_get_num_inputs (UfoTask *task)
{
    return 1;
}

static guint
ufo_filter_task_get_num_dimensions (UfoTask *task,
                                    guint input)
{
    g_return_val_if_fail (input == 0, 0);
    // introduced "Stack" filter before "backproject", the third dimension indicates num of sinograms in a stream stacked together
    return 3;
}

static UfoTaskMode
ufo_filter_task_get_mode (UfoTask *task)
{
    return UFO_TASK_MODE_PROCESSOR | UFO_TASK_MODE_GPU;
}

static gboolean
ufo_backproject_task_equal_real (UfoNode *n1,
                                 UfoNode *n2)
{
    g_return_val_if_fail (UFO_IS_BACKPROJECT_TASK (n1) && UFO_IS_BACKPROJECT_TASK (n2), FALSE);
    return UFO_BACKPROJECT_TASK (n1)->priv->texture_kernel == UFO_BACKPROJECT_TASK (n2)->priv->texture_kernel;
}

static void
ufo_backproject_task_finalize (GObject *object)
{
    UfoBackprojectTaskPrivate *priv;

    priv = UFO_BACKPROJECT_TASK_GET_PRIVATE (object);

    release_lut_mems (priv);

    g_free (priv->host_sin_lut);
    g_free (priv->host_cos_lut);

    if (priv->nearest_kernel) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->nearest_kernel));
        priv->nearest_kernel = NULL;
    }

    if (priv->texture_kernel) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->texture_kernel));
        priv->texture_kernel = NULL;
    }

    if (priv->optimized_kernel_2d) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->optimized_kernel_2d));
        priv->optimized_kernel_2d = NULL;
    }

    if (priv->optimized_kernel_3d) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->optimized_kernel_3d));
        priv->optimized_kernel_3d = NULL;
    }

    if (priv->interleave) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->interleave));
        priv->interleave = NULL;
    }

    if (priv->uninterleave) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->uninterleave));
        priv->uninterleave = NULL;
    }

    if (priv->context) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseContext (priv->context));
        priv->context = NULL;
    }

    G_OBJECT_CLASS (ufo_backproject_task_parent_class)->finalize (object);
}

static void
ufo_task_interface_init (UfoTaskIface *iface)
{
    iface->setup = ufo_backproject_task_setup;
    iface->get_requisition = ufo_backproject_task_get_requisition;
    iface->get_num_inputs = ufo_filter_task_get_num_inputs;
    iface->get_num_dimensions = ufo_filter_task_get_num_dimensions;
    iface->get_mode = ufo_filter_task_get_mode;
    iface->process = ufo_backproject_task_process;
}

static void
ufo_backproject_task_set_property (GObject *object,
                                   guint property_id,
                                   const GValue *value,
                                   GParamSpec *pspec)
{
    UfoBackprojectTaskPrivate *priv = UFO_BACKPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_NUM_PROJECTIONS:
            priv->n_projections = g_value_get_uint (value);
            break;
        case PROP_OFFSET:
            priv->offset = g_value_get_uint (value);
            break;
        case PROP_AXIS_POSITION:
            priv->axis_pos = g_value_get_double (value);
            break;
        case PROP_ANGLE_STEP:
            priv->angle_step = g_value_get_double (value);
            break;
        case PROP_ANGLE_OFFSET:
            priv->angle_offset = g_value_get_double (value);
            priv->luts_changed = TRUE;
            break;
        case PROP_MODE:
            priv->mode = g_value_get_enum (value);
            break;
        case PROP_ROI_X:
            priv->roi_x = g_value_get_uint (value);
            break;
        case PROP_ROI_Y:
            priv->roi_y = g_value_get_uint (value);
            break;
        case PROP_ROI_WIDTH:
            priv->roi_width = g_value_get_uint (value);
            break;
        case PROP_ROI_HEIGHT:
            priv->roi_height = g_value_get_uint (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_backproject_task_get_property (GObject *object,
                                   guint property_id,
                                   GValue *value,
                                   GParamSpec *pspec)
{
    UfoBackprojectTaskPrivate *priv = UFO_BACKPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_NUM_PROJECTIONS:
            g_value_set_uint (value, priv->n_projections);
            break;
        case PROP_OFFSET:
            g_value_set_uint (value, priv->offset);
            break;
        case PROP_AXIS_POSITION:
            g_value_set_double (value, priv->axis_pos);
            break;
        case PROP_ANGLE_STEP:
            g_value_set_double (value, priv->angle_step);
            break;
        case PROP_ANGLE_OFFSET:
            g_value_set_double (value, priv->angle_offset);
            break;
        case PROP_MODE:
            g_value_set_enum (value, priv->mode);
            break;
        case PROP_ROI_X:
            g_value_set_uint (value, priv->roi_x);
            break;
        case PROP_ROI_Y:
            g_value_set_uint (value, priv->roi_y);
            break;
        case PROP_ROI_WIDTH:
            g_value_set_uint (value, priv->roi_width);
            break;
        case PROP_ROI_HEIGHT:
            g_value_set_uint (value, priv->roi_height);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void
ufo_backproject_task_class_init (UfoBackprojectTaskClass *klass)
{
    GObjectClass *oclass;
    UfoNodeClass *node_class;

    oclass = G_OBJECT_CLASS (klass);
    node_class = UFO_NODE_CLASS (klass);

    oclass->finalize = ufo_backproject_task_finalize;
    oclass->set_property = ufo_backproject_task_set_property;
    oclass->get_property = ufo_backproject_task_get_property;

    properties[PROP_NUM_PROJECTIONS] =
            g_param_spec_uint ("num-projections",
                               "Number of projections between 0 and 180 degrees",
                               "Number of projections between 0 and 180 degrees",
                               0, +32768, 0,
                               G_PARAM_READWRITE);

    properties[PROP_OFFSET] =
            g_param_spec_uint ("offset",
                               "Offset to the first projection",
                               "Offset to the first projection",
                               0, +32768, 0,
                               G_PARAM_READWRITE);

    properties[PROP_AXIS_POSITION] =
            g_param_spec_double ("axis-pos",
                                 "Position of rotation axis",
                                 "Position of rotation axis",
                                 -1.0, +32768.0, 0.0,
                                 G_PARAM_READWRITE);

    properties[PROP_ANGLE_STEP] =
            g_param_spec_double ("angle-step",
                                 "Increment of angle in radians",
                                 "Increment of angle in radians",
                                 -G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
                                 G_PARAM_READWRITE);

    properties[PROP_ANGLE_OFFSET] =
            g_param_spec_double ("angle-offset",
                                 "Angle offset in radians",
                                 "Angle offset in radians determining the first angle position",
                                 0.0, G_MAXDOUBLE, 0.0,
                                 G_PARAM_READWRITE);

    properties[PROP_MODE] =
            g_param_spec_enum ("mode",
                               "Backprojection mode (\"nearest\", \"texture\")",
                               "Backprojection mode (\"nearest\", \"texture\")",
                               g_enum_register_static ("ufo_backproject_mode", mode_values),
                               MODE_TEXTURE, G_PARAM_READWRITE);

    properties[PROP_ROI_X] =
            g_param_spec_uint ("roi-x",
                               "X coordinate of region of interest",
                               "X coordinate of region of interest",
                               0, G_MAXUINT, 0,
                               G_PARAM_READWRITE);

    properties[PROP_ROI_Y] =
            g_param_spec_uint ("roi-y",
                               "Y coordinate of region of interest",
                               "Y coordinate of region of interest",
                               0, G_MAXUINT, 0,
                               G_PARAM_READWRITE);

    properties[PROP_ROI_WIDTH] =
            g_param_spec_uint ("roi-width",
                               "Width of region of interest",
                               "Width of region of interest",
                               0, G_MAXUINT, 0,
                               G_PARAM_READWRITE);

    properties[PROP_ROI_HEIGHT] =
            g_param_spec_uint ("roi-height",
                               "Height of region of interest",
                               "Height of region of interest",
                               0, G_MAXUINT, 0,
                               G_PARAM_READWRITE);

    for (guint i = PROP_0 + 1; i < N_PROPERTIES; i++)
        g_object_class_install_property (oclass, i, properties[i]);

    node_class->equal = ufo_backproject_task_equal_real;

    g_type_class_add_private(klass, sizeof(UfoBackprojectTaskPrivate));
}

static void
ufo_backproject_task_init (UfoBackprojectTask *self)
{
    UfoBackprojectTaskPrivate *priv;
    self->priv = priv = UFO_BACKPROJECT_TASK_GET_PRIVATE (self);
    priv->nearest_kernel = NULL;
    priv->texture_kernel = NULL;
    priv->optimized_kernel_2d = NULL;
    priv->interleave = NULL;
    priv->uninterleave = NULL;
    priv->optimized_kernel_3d = NULL;
    priv->n_projections = 0;
    priv->offset = 0;
    priv->axis_pos = -1.0;
    priv->angle_step = -1.0;
    priv->angle_offset = 0.0;
    priv->real_angle_step = -1.0;
    priv->sin_lut = NULL;
    priv->cos_lut = NULL;
    priv->host_sin_lut = NULL;
    priv->host_cos_lut = NULL;
    priv->mode = MODE_TEXTURE;
    priv->luts_changed = TRUE;
    priv->roi_x = priv->roi_y = 0;
    priv->roi_width = priv->roi_height = 0;
}