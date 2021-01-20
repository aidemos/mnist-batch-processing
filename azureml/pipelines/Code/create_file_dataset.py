import argparse
from azureml.core import Dataset, Run
from azureml.core.dataset import Dataset
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.pipeline.core import PublishedPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--process_folder_param", type=str, help="process folder path")
args = parser.parse_args()

run = Run.get_context()
ws = run.experiment.workspace

def_data_store = ws.get_default_datastore()
mnist_ds_name = 'mnist_version_10_ds_'+ args.process_folder_param
print(mnist_ds_name)

path_on_datastore = def_data_store.path(args.process_folder_param)
input_mnist_ds = Dataset.File.from_files(path=path_on_datastore, validate=False)

# input_mnist_ds = input_mnist_ds.register(workspace=ws,
#                                  name= mnist_ds_name,
#                                  description='mnist images')

experiment = Experiment(ws, 'digit_identification')
published_pipeline = PublishedPipeline.get(workspace=ws, id="880353cd-1950-425a-b54b-1f89b625413e")
pipeline_run = experiment.submit(published_pipeline, 
                                   pipeline_parameters={"mnist_param": input_mnist_ds})