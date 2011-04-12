#include <glib.h>
#include <ethos/ethos.h>
#include <json-glib/json-glib.h>

#include "ufo-graph.h"
#include "ufo-connection.h"
#include "ufo-container.h"
#include "ufo-sequence.h"
#include "ufo-split.h"

G_DEFINE_TYPE(UfoGraph, ufo_graph, G_TYPE_OBJECT);

#define UFO_GRAPH_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_GRAPH, UfoGraphPrivate))

struct _UfoGraphPrivate {
    EthosManager        *ethos;
    UfoResourceManager  *resource_manager;
    UfoContainer        *root_container;
    GHashTable          *graph;     /**< maps from UfoFilter* to UfoConnection* */
    GHashTable          *plugins;   /**< maps from gchar* to EthosPlugin* */
};

GList *ufo_graph_get_filter_names(UfoGraph *self)
{
    return g_hash_table_get_keys(self->priv->plugins);
}

UfoFilter *ufo_graph_create_node(UfoGraph *self, gchar *filter_name)
{
    EthosPlugin *plugin = g_hash_table_lookup(self->priv->plugins, filter_name);
    /* TODO: When we move to libpeas we have to instantiate new objects each
     * time a user requests a new stateful node. */
    if (plugin != NULL) {
        UfoFilter *filter = (UfoFilter *) plugin;
        ufo_filter_set_resource_manager(filter, self->priv->resource_manager);
        return filter;
    }
    return NULL;
}

void ufo_graph_connect(UfoGraph *self, UfoFilter *src, UfoFilter *dst)
{
    UfoConnection *connection = ufo_connection_new();
    /*ufo_connection_set_filters(connection, src, dst);*/

    GAsyncQueue *queue = ufo_connection_get_queue(connection);
    ufo_filter_set_output_queue(src, queue);
    ufo_filter_set_input_queue(dst, queue);

    g_hash_table_replace(self->priv->graph, src, connection);
    g_hash_table_replace(self->priv->graph, dst, NULL);
}

static void ufo_build_graph(UfoGraph *self, JsonNode *node)
{
    /* We look for a sequence, split or filter node and add those recursively. */    

}

void ufo_graph_read_json_configuration(UfoGraph *self, GString *filename)
{
    static const char *config = 
        "{"
        "  \"properties\" : { \"foo\" : 42 },"
        "  \"sequence\" : ["
        "     { \"filter\" : \"file-reader\" },"
        "     { \"filter\" : \"noise-reduction\" }"
        "  ]"
        "}\0";

    JsonParser *parser = json_parser_new();
    GError *error = NULL;
    json_parser_load_from_data(parser, config, -1, &error);
    if (error) {
        g_error_free(error);
        g_object_unref(parser);
        return;
    }

    JsonNode *root = json_parser_get_root(parser);

    JsonObject *object = json_node_get_object(root);
    GList *children = json_object_get_members(object);
    for (guint i = 0; i < g_list_length(children); i++) {
        const char *name = (const char *) g_list_nth_data(children, i);
        if (g_strcmp0(name, "sequence") == 0) {
            self->priv->root_container = (UfoContainer *) ufo_sequence_new();
        }
        else if (g_strcmp0(name, "split") == 0) {
            self->priv->root_container = (UfoContainer *) ufo_split_new();
        }
    }

    ufo_build_graph(self, root);
    g_object_unref(parser);
}

void ufo_graph_run(UfoGraph *self)
{
}

UfoGraph *ufo_graph_new()
{
    return g_object_new(UFO_TYPE_GRAPH, NULL);
}

static void ufo_graph_dispose(GObject *gobject)
{
    UfoGraph *self = UFO_GRAPH(gobject);
    
    if (self->priv->graph) {
        g_hash_table_destroy(self->priv->graph);
        self->priv->graph = NULL;
    }

    if (self->priv->plugins) {
        g_hash_table_destroy(self->priv->plugins);
        self->priv->plugins = NULL;
    }

    if (self->priv->resource_manager) {
        g_object_unref(self->priv->resource_manager);
        self->priv->resource_manager = NULL;
    }

    G_OBJECT_CLASS(ufo_graph_parent_class)->dispose(gobject);
}

static void ufo_graph_class_init(UfoGraphClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);

    gobject_class->dispose = ufo_graph_dispose;

    /* install private data */
    g_type_class_add_private(klass, sizeof(UfoGraphPrivate));
}

static void ufo_graph_add_plugin(gpointer data, gpointer user_data)
{
    EthosPluginInfo *info = (EthosPluginInfo *) data;
    UfoGraphPrivate *priv = (UfoGraphPrivate *) user_data;

    g_message("Load filter: %s", ethos_plugin_info_get_name(info));

    g_hash_table_insert(priv->plugins, 
        (gpointer) ethos_plugin_info_get_name(info),
        ethos_manager_get_plugin(priv->ethos, info));
}

static void ufo_graph_init(UfoGraph *self)
{
    /* init public fields */

    /* init private fields */
    UfoGraphPrivate *priv;
    self->priv = priv = UFO_GRAPH_GET_PRIVATE(self);

    /* TODO: determine directories in a better way */
    gchar *plugin_dirs[] = { "/usr/local/lib", "../filters", NULL };

    priv->ethos = ethos_manager_new_full("UFO", plugin_dirs);
    ethos_manager_initialize(priv->ethos);

    priv->plugins = g_hash_table_new(g_str_hash, g_str_equal);
    GList *plugin_info = ethos_manager_get_plugin_info(priv->ethos);

    g_list_foreach(plugin_info, &ufo_graph_add_plugin, priv);
    g_list_free(plugin_info);

    priv->graph = g_hash_table_new(NULL, NULL);
    priv->resource_manager = ufo_resource_manager_new();
    priv->root_container = NULL;
}


