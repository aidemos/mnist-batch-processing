
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from azureml.core import Model
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
from azureml.core import Run

run = Run.get_context(allow_offline=False)

custom_dimensions = {
    "parent_run_id": run.parent.id,
    "step_id": run.id,
    "step_name": run.name,
    "experiment_name": run.experiment.name,
    "run_url": run.parent.get_portal_url(),
    "run_type": "inference"
}

def init():
    global g_tf_sess

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    # Assumes the environment variable APPLICATIONINSIGHTS_CONNECTION_STRING is already set
    logger.addHandler(AzureLogHandler())
    #logger.warning("Processing image")
    # Assumes AzureLogHandler was already registered above
    logger.info("Inferring image", extra= {"custom_dimensions":custom_dimensions})
    # pull down model from workspace
    model_path = Model.get_model_path("mnist-prs")

    # contruct graph to execute
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(model_path, 'mnist-tf.model.meta'))
    g_tf_sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    saver.restore(g_tf_sess, os.path.join(model_path, 'mnist-tf.model'))


def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []
    in_tensor = g_tf_sess.graph.get_tensor_by_name("network/X:0")
    output = g_tf_sess.graph.get_tensor_by_name("network/output/MatMul:0")

    for image in mini_batch:
        #check file format
        filename, file_extension = os.path.splitext(image)
        if file_extension==".png":
            # prepare each image
            data = Image.open(image)
            np_im = np.array(data).reshape((1, 784))
            # perform inference
            inference_result = output.eval(feed_dict={in_tensor: np_im}, session=g_tf_sess)
            # find best probability, and add to result list
            best_result = np.argmax(inference_result)
            resultList.append("{}: {}".format(os.path.basename(image), best_result))
    return resultList
