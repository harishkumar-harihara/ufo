#include <gmodule.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <math.h>
#include <ufo/ufo-resource-manager.h>
#include <ufo/ufo-filter.h>
#include <ufo/ufo-buffer.h>
#include "ufo-filter-gaussian-blur.h"

/**
 * SECTION:ufo-filter-gaussian-blur
 * @Short_description:
 * @Title: gaussianblur
 *
 * Detailed description.
 */

struct _UfoFilterGaussianBlurPrivate {
    guint size;
    gfloat sigma;
    cl_kernel h_kernel;
    cl_kernel v_kernel;
};

G_DEFINE_TYPE(UfoFilterGaussianBlur, ufo_filter_gaussian_blur, UFO_TYPE_FILTER)

#define UFO_FILTER_GAUSSIAN_BLUR_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_FILTER_GAUSSIAN_BLUR, UfoFilterGaussianBlurPrivate))

enum {
    PROP_0,
    PROP_SIZE,
    PROP_SIGMA,
    N_PROPERTIES
};

static GParamSpec *gaussian_blur_properties[N_PROPERTIES] = { NULL, };


static void ufo_filter_gaussian_blur_initialize(UfoFilter *filter)
{
    UfoFilterGaussianBlur *self = UFO_FILTER_GAUSSIAN_BLUR(filter);
    UfoResourceManager *manager = ufo_resource_manager();
    GError *error = NULL;
    self->priv->h_kernel = ufo_resource_manager_get_kernel(manager, "gaussian.cl", "h_gaussian", &error);
    self->priv->v_kernel = ufo_resource_manager_get_kernel(manager, "gaussian.cl", "v_gaussian", &error);

    if (error != NULL) {
        g_warning("%s", error->message);
        g_error_free(error);
    }
}

/*
 * This is the main method in which the filter processes one buffer after
 * another.
 */
static void ufo_filter_gaussian_blur_process(UfoFilter *filter)
{
    g_return_if_fail(UFO_IS_FILTER(filter));
    UfoFilterGaussianBlurPrivate *priv = UFO_FILTER_GAUSSIAN_BLUR_GET_PRIVATE(filter);
    UfoChannel *input_channel = ufo_filter_get_input_channel(filter);
    UfoChannel *output_channel = ufo_filter_get_output_channel(filter);
    UfoBuffer *input = ufo_channel_get_input_buffer(input_channel);
    UfoBuffer *output = NULL;
    UfoResourceManager *manager = ufo_resource_manager();
    cl_context context = (cl_context) ufo_resource_manager_get_context(manager);
    cl_command_queue command_queue = (cl_command_queue) ufo_filter_get_command_queue(filter);
    cl_int error = CL_SUCCESS;

    if (input == NULL) {
        ufo_channel_finish(output_channel);
        return;
    }

    guint width, height;
    ufo_buffer_get_2d_dimensions(input, &width, &height);
    ufo_channel_allocate_output_buffers_like(output_channel, input);

    const guint kernel_size = priv->size;
    const guint half_kernel_size = kernel_size / 2;
    gfloat *weights = g_malloc0(kernel_size * sizeof(gfloat));
    gfloat weight_sum = 0.0;
    
    for (guint i = 0; i < half_kernel_size + 1; i++) {
        gfloat x = (gfloat) half_kernel_size - i;
        weights[i] = (gfloat) 1.0 / (priv->sigma * sqrt(2*G_PI)) * exp((x * x) / (-2.0 * priv->sigma * priv->sigma));
        weights[kernel_size-i-1] = weights[i];
    }

    for (guint i = 0; i < kernel_size; i++)
        weight_sum += weights[i];

    for (guint i = 0; i < kernel_size; i++)
        weights[i] /= weight_sum;

    cl_mem weights_mem = clCreateBuffer(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            5 * sizeof(float), weights, &error);

    CHECK_OPENCL_ERROR(error);

    cl_mem intermediate_mem = clCreateBuffer(context,
            CL_MEM_READ_WRITE, width * height * sizeof(float), NULL, &error);

    CHECK_OPENCL_ERROR(error);

    CHECK_OPENCL_ERROR(clSetKernelArg(priv->h_kernel, 2, sizeof(cl_mem), &weights_mem));
    CHECK_OPENCL_ERROR(clSetKernelArg(priv->h_kernel, 3, sizeof(int), &half_kernel_size));
    CHECK_OPENCL_ERROR(clSetKernelArg(priv->v_kernel, 2, sizeof(cl_mem), &weights_mem));
    CHECK_OPENCL_ERROR(clSetKernelArg(priv->v_kernel, 3, sizeof(int), &half_kernel_size));

    const size_t global_work_size[] = { width, height };

    while (input != NULL) {
        output = ufo_channel_get_output_buffer(output_channel);
        cl_mem input_mem = ufo_buffer_get_device_array(input, command_queue);
        cl_mem output_mem = ufo_buffer_get_device_array(output, command_queue);

        CHECK_OPENCL_ERROR(clSetKernelArg(priv->h_kernel, 0, sizeof(cl_mem), &input_mem));
        CHECK_OPENCL_ERROR(clSetKernelArg(priv->h_kernel, 1, sizeof(cl_mem), &intermediate_mem));

        CHECK_OPENCL_ERROR(clEnqueueNDRangeKernel(command_queue, priv->h_kernel,
                    2, NULL, global_work_size, NULL,
                    0, NULL, NULL));

        CHECK_OPENCL_ERROR(clSetKernelArg(priv->v_kernel, 0, sizeof(cl_mem), &intermediate_mem));
        CHECK_OPENCL_ERROR(clSetKernelArg(priv->v_kernel, 1, sizeof(cl_mem), &output_mem));

        CHECK_OPENCL_ERROR(clEnqueueNDRangeKernel(command_queue, priv->v_kernel,
                    2, NULL, global_work_size, NULL,
                    0, NULL, NULL));

        ufo_channel_finalize_input_buffer(input_channel, input);
        ufo_channel_finalize_output_buffer(output_channel, output);
        input = ufo_channel_get_input_buffer(input_channel);
    }

    CHECK_OPENCL_ERROR(clReleaseMemObject(weights_mem)); 
    CHECK_OPENCL_ERROR(clReleaseMemObject(intermediate_mem)); 
    g_free(weights);
    ufo_channel_finish(output_channel);
}

static void ufo_filter_gaussian_blur_set_property(GObject *object,
    guint           property_id,
    const GValue    *value,
    GParamSpec      *pspec)
{
    UfoFilterGaussianBlurPrivate *priv = UFO_FILTER_GAUSSIAN_BLUR_GET_PRIVATE(object);

    switch (property_id) {
        case PROP_SIZE:
            priv->size = g_value_get_uint(value);
            break;
        case PROP_SIGMA:
            priv->sigma = g_value_get_float(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void ufo_filter_gaussian_blur_get_property(GObject *object,
    guint       property_id,
    GValue      *value,
    GParamSpec  *pspec)
{
    UfoFilterGaussianBlurPrivate *priv = UFO_FILTER_GAUSSIAN_BLUR_GET_PRIVATE(object);

    switch (property_id) {
        case PROP_SIZE:
            g_value_set_uint(value, priv->size);
            break;
        case PROP_SIGMA:
            g_value_set_float(value, priv->sigma);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void ufo_filter_gaussian_blur_class_init(UfoFilterGaussianBlurClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    UfoFilterClass *filter_class = UFO_FILTER_CLASS(klass);

    gobject_class->set_property = ufo_filter_gaussian_blur_set_property;
    gobject_class->get_property = ufo_filter_gaussian_blur_get_property;
    filter_class->initialize = ufo_filter_gaussian_blur_initialize;
    filter_class->process = ufo_filter_gaussian_blur_process;

    gaussian_blur_properties[PROP_SIZE] = 
        g_param_spec_uint("size",
            "Size of the kernel",
            "Size of the kernel",
            3, 1000, 5,
            G_PARAM_READWRITE);

    gaussian_blur_properties[PROP_SIGMA] = 
        g_param_spec_float("sigma",
            "sigma",
            "sigma",
            1.0f, 1000.0f, 1.0f,
            G_PARAM_READWRITE);

    g_object_class_install_property(gobject_class, PROP_SIZE, gaussian_blur_properties[PROP_SIZE]);
    g_object_class_install_property(gobject_class, PROP_SIGMA, gaussian_blur_properties[PROP_SIGMA]);

    g_type_class_add_private(gobject_class, sizeof(UfoFilterGaussianBlurPrivate));
}

static void ufo_filter_gaussian_blur_init(UfoFilterGaussianBlur *self)
{
    UfoFilterGaussianBlurPrivate *priv = self->priv = UFO_FILTER_GAUSSIAN_BLUR_GET_PRIVATE(self);

    priv->size = 5;
    priv->sigma = 1.0f;
    ufo_filter_register_input(UFO_FILTER(self), "input", 2);
    ufo_filter_register_output(UFO_FILTER(self), "output", 2);
}

G_MODULE_EXPORT UfoFilter *ufo_filter_plugin_new(void)
{
    return g_object_new(UFO_TYPE_FILTER_GAUSSIAN_BLUR, NULL);
}
