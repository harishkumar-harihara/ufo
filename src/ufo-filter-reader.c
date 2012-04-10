#include <gmodule.h>
#include <stdlib.h>
#include <tiffio.h>
#include <glob.h>

#include <ufo/ufo-filter.h>
#include <ufo/ufo-buffer.h>
#include <ufo/ufo-resource-manager.h>

#include "ufo-filter-reader.h"

/**
 * SECTION:ufo-filter-reader
 * @Short_description: Read TIFF and EDF files
 * @Title: reader 
 *
 * The reader node loads single files from disk and provides them as a stream in
 * output "image".
 */

struct _UfoFilterReaderPrivate {
    gchar *path;
    gint count;
    gint nth;
    gboolean blocking;
    gboolean normalize;
};

G_DEFINE_TYPE(UfoFilterReader, ufo_filter_reader, UFO_TYPE_FILTER)

#define UFO_FILTER_READER_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_FILTER_READER, UfoFilterReaderPrivate))

enum {
    PROP_0,
    PROP_PATH,
    PROP_COUNT,
    PROP_BLOCKING,
    PROP_NTH,
    PROP_NORMALIZE,
    N_PROPERTIES
};

static GParamSpec *reader_properties[N_PROPERTIES] = { NULL, };

static gpointer image_buffer = NULL;
static gsize image_size = 0;

static gboolean filter_decode_tiff(TIFF *tif, void *buffer)
{
    const tsize_t strip_size = TIFFStripSize(tif);
    const guint n_strips = TIFFNumberOfStrips(tif);
    int offset = 0;
    tsize_t result = 0;

    for (guint strip = 0; strip < n_strips; strip++) {
        result = TIFFReadEncodedStrip(tif, strip, ((gchar *) buffer) +offset, strip_size);
        if (result == -1)
            return FALSE;
        offset += result;
    }
    return TRUE;
}

/**
 * filter_read_tiff:
 *
 * Returns: TRUE if more frames can be read from @tif
 */
static gboolean filter_read_tiff(TIFF *tif,
    gpointer *buffer,
    guint16 *bytes_per_sample,
    guint16 *samples_per_pixel,
    guint32 *width,
    guint32 *height)
{
    guint16 bits_per_sample = 8;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);

    if (*samples_per_pixel > 1)
        g_warning("TIFF has %i samples per pixel", *samples_per_pixel);

    *bytes_per_sample = bits_per_sample >> 3;
    gsize size = *bytes_per_sample * (*width) * (*height);

    if ((*buffer == NULL) || (image_size != size)) {
        *buffer = g_malloc0(size); 
        image_size = size;
    }

    if (!filter_decode_tiff(tif, *buffer))
        goto error_close;

    return TIFFReadDirectory(tif) == 1;

error_close:
    *buffer = NULL;
    return FALSE;
}

static void *filter_read_edf(const gchar *filename, 
    guint16 *bytes_per_sample,
    guint16 *samples_per_pixel,
    guint32 *width, 
    guint32 *height)
{
    FILE *fp = fopen(filename, "rb");  
    gchar *header = g_malloc(1024);

    size_t num_bytes = fread(header, 1, 1024, fp);
    if (num_bytes != 1024) {
        g_free(header);
        fclose(fp);
        return NULL;
    }

    gchar **tokens = g_strsplit(header, ";", 0);
    gboolean big_endian = FALSE;
    int index = 0;
    guint w = 0, h = 0, size = 0;

    while (tokens[index] != NULL) {
        gchar **key_value = g_strsplit(tokens[index], "=", 0);
        if (g_strcmp0(g_strstrip(key_value[0]), "Dim_1") == 0)
            w = (guint) atoi(key_value[1]);
        else if (g_strcmp0(g_strstrip(key_value[0]), "Dim_2") == 0)
            h = (guint) atoi(key_value[1]);
        else if (g_strcmp0(g_strstrip(key_value[0]), "Size") == 0)
            size = (guint) atoi(key_value[1]);
        else if ((g_strcmp0(g_strstrip(key_value[0]), "ByteOrder") == 0) &&
                 (g_strcmp0(g_strstrip(key_value[1]), "HighByteFirst") == 0))
            big_endian = TRUE;
        g_strfreev(key_value);
        index++;
    }

    g_strfreev(tokens);
    g_free(header);

    if (w * h * sizeof(float) != size) {
        fclose(fp);
        return NULL;
    }

    *bytes_per_sample = 4;
    *samples_per_pixel = 1;
    *width = w;
    *height = h;

    /* Skip header */
    fseek(fp, 0L, SEEK_END);
    gssize file_size = (gssize) ftell(fp);
    fseek(fp, file_size - size, SEEK_SET);

    /* Read data */
    if ((image_buffer == NULL) || (image_size != size)) {
        image_buffer = g_malloc(size);
        image_size = size;
    }

    num_bytes = fread(image_buffer, 1, size, fp);
    fclose(fp);
    if (num_bytes != size) {
        g_free(image_buffer);
        return NULL;
    }

    if ((G_BYTE_ORDER == G_LITTLE_ENDIAN) && big_endian) {
        guint32 *data = (guint32 *) image_buffer;    
        for (guint i = 0; i < w*h; i++)
            data[i] = g_ntohl(data[i]);
    }
    return image_buffer;
}

static void filter_dispose_filenames(GSList *filenames)
{
    if (filenames != NULL) {
        g_slist_foreach(filenames, (GFunc) g_free, NULL);
        filenames = NULL;
    }
}

static GSList *filter_read_filenames(UfoFilterReaderPrivate *priv)
{
    GSList *result = NULL;
    glob_t glob_vector;
    guint i = (priv->nth < 0) ? 0 : (guint) priv->nth;
    glob(priv->path, GLOB_MARK | GLOB_TILDE, NULL, &glob_vector);

    while (i < glob_vector.gl_pathc)
        result = g_slist_append(result, g_strdup(glob_vector.gl_pathv[i++]));

    globfree(&glob_vector);
    return result;
}

static void filter_push_data(UfoChannel *output_channel, void *buffer, 
        guint width, guint height, guint bytes_per_sample, gboolean normalize)
{
    UfoBuffer *output_buffer = ufo_channel_get_output_buffer(output_channel);
    ufo_buffer_set_host_array(output_buffer, buffer, bytes_per_sample * width * height, NULL);

    if (bytes_per_sample < 4)
        ufo_buffer_reinterpret(output_buffer, bytes_per_sample << 3, width * height, normalize);

    ufo_channel_finalize_output_buffer(output_channel, output_buffer);
}

static void ufo_filter_reader_process(UfoFilter *self)
{
    g_return_if_fail(UFO_IS_FILTER(self));

    UfoFilterReaderPrivate *priv = UFO_FILTER_READER_GET_PRIVATE(self);
    UfoChannel *output_channel = ufo_filter_get_output_channel(self);
    
    GSList *filenames = filter_read_filenames(priv);
    GSList *filename = filenames;
    guint count = priv->count == -1 ? G_MAXUINT : (guint) priv->count;
    guint current_frame = 0;
    guint src_width, src_height;
    guint16 bytes_per_sample, samples_per_pixel;
    gboolean normalize = priv->normalize;
    gboolean buffers_initialized = FALSE;
    gpointer frame_buffer = NULL;

    while ((filename != NULL) && (current_frame < count)) {
        if (g_str_has_suffix(filename->data, "tif")) {;
            TIFF *tif = TIFFOpen(filename->data, "r");

            if (tif == NULL)
                break;

            gboolean more_pages = TRUE;

            while ((current_frame < count) && more_pages) {
                more_pages = filter_read_tiff(tif, &frame_buffer,
                    &bytes_per_sample, &samples_per_pixel, &src_width, &src_height);

                if (frame_buffer == NULL)
                    break;

                if (!buffers_initialized) {
                    guint dimensions[2] = { src_width, src_height };
                    ufo_channel_allocate_output_buffers(output_channel, 2, dimensions);
                    buffers_initialized = TRUE;
                }

                filter_push_data(output_channel, frame_buffer, src_width, src_height, bytes_per_sample, normalize);
                current_frame++;
            }
        }
        else {
            frame_buffer = filter_read_edf((char *) filename->data,
                    &bytes_per_sample, &samples_per_pixel,
                    &src_width, &src_height);

            if (frame_buffer == NULL)
                break;

            filter_push_data(output_channel, frame_buffer, src_width, src_height, bytes_per_sample, normalize);
            current_frame++;
        }

        filename = g_slist_next(filename);
    }

    ufo_channel_finish(output_channel);
    filter_dispose_filenames(filenames);
    g_free(frame_buffer);
}

static void ufo_filter_reader_set_property(GObject *object,
    guint           property_id,
    const GValue    *value,
    GParamSpec      *pspec)
{
    UfoFilterReaderPrivate *priv = UFO_FILTER_READER_GET_PRIVATE(object);

    switch (property_id) {
        case PROP_PATH:
            g_free(priv->path);
            priv->path = g_value_dup_string(value);
            break;
        case PROP_COUNT:
            priv->count = g_value_get_int(value);
            break;
        case PROP_NTH:
            priv->nth = g_value_get_int(value);
            break;
        case PROP_BLOCKING:
            priv->blocking = g_value_get_boolean(value);
            break;
        case PROP_NORMALIZE:
            priv->normalize = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void ufo_filter_reader_get_property(GObject *object,
    guint       property_id,
    GValue      *value,
    GParamSpec  *pspec)
{
    UfoFilterReaderPrivate *priv = UFO_FILTER_READER_GET_PRIVATE(object);

    switch (property_id) {
        case PROP_PATH:
            g_value_set_string(value, priv->path);
            break;
        case PROP_COUNT:
            g_value_set_int(value, priv->count);
            break;
        case PROP_NTH:
            g_value_set_int(value, priv->nth);
            break;
        case PROP_BLOCKING:
            g_value_set_boolean(value, priv->blocking);
            break;
        case PROP_NORMALIZE:
            g_value_set_boolean(value, priv->normalize);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
            break;
    }
}

static void ufo_filter_reader_finalize(GObject *object)
{
    UfoFilterReaderPrivate *priv = UFO_FILTER_READER_GET_PRIVATE(object);

    if (priv->path)
        g_free(priv->path);

    if (image_buffer != NULL)
        g_free(image_buffer);

    G_OBJECT_CLASS(ufo_filter_reader_parent_class)->finalize(object);
}

static void ufo_filter_reader_class_init(UfoFilterReaderClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    UfoFilterClass *filter_class = UFO_FILTER_CLASS(klass);

    gobject_class->set_property = ufo_filter_reader_set_property;
    gobject_class->get_property = ufo_filter_reader_get_property;
    gobject_class->finalize = ufo_filter_reader_finalize;
    filter_class->process = ufo_filter_reader_process;

    reader_properties[PROP_PATH] = 
        g_param_spec_string("path",
            "Glob-style pattern.",
            "Glob-style pattern that describes the file path.",
            "*.tif",
            G_PARAM_READWRITE);

    reader_properties[PROP_COUNT] =
        g_param_spec_int("count",
        "Number of files",
        "Number of files to read.",
        -1, G_MAXINT, -1,
        G_PARAM_READWRITE);

    reader_properties[PROP_NTH] =
        g_param_spec_int("nth",
        "Read from nth file",
        "Read from nth file.",
        -1, G_MAXINT, -1,
        G_PARAM_READWRITE);

    /**
     * UfoFilterReader:blocking:
     *
     * Block the reader and do not return unless #UfoFilterReader:count files
     * have been read. This is useful in case not all files are available the
     * time the reader was started.
     */
    reader_properties[PROP_BLOCKING] = 
        g_param_spec_boolean("blocking",
        "Block reader",
        "Block until all files are read.",
        FALSE,
        G_PARAM_READWRITE);

    reader_properties[PROP_NORMALIZE] = 
        g_param_spec_boolean("normalize",
        "Normalize values",
        "Whether 8-bit or 16-bit values are normalized to [0.0, 1.0]",
        FALSE,
        G_PARAM_READWRITE);

    g_object_class_install_property(gobject_class, PROP_PATH, reader_properties[PROP_PATH]);
    g_object_class_install_property(gobject_class, PROP_COUNT, reader_properties[PROP_COUNT]);
    g_object_class_install_property(gobject_class, PROP_NTH, reader_properties[PROP_NTH]);
    g_object_class_install_property(gobject_class, PROP_BLOCKING, reader_properties[PROP_BLOCKING]);
    g_object_class_install_property(gobject_class, PROP_NORMALIZE, reader_properties[PROP_NORMALIZE]);

    g_type_class_add_private(gobject_class, sizeof(UfoFilterReaderPrivate));
}

static void ufo_filter_reader_init(UfoFilterReader *self)
{
    UfoFilterReaderPrivate *priv = NULL;
    self->priv = priv = UFO_FILTER_READER_GET_PRIVATE(self);
    priv->path = g_strdup("*.tif");
    priv->count = -1;
    priv->nth = -1;
    priv->blocking = FALSE;
    priv->normalize = FALSE;

    ufo_filter_register_output(UFO_FILTER(self), "image", 2);
}

G_MODULE_EXPORT UfoFilter *ufo_filter_plugin_new(void)
{
    return g_object_new(UFO_TYPE_FILTER_READER, NULL);
}
