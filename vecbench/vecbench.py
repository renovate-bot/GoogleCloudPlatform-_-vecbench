# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Main for the vector benchmarking project.
"""
import os
import argparse
import logging

from pyaml_env import parse_config
from datasets.util import make_hdf5_file
from mp.vecbenchloader import Loader
from report.report import generate_report
from experiments.experiment import Experiment
import metrics

logging.getLogger().setLevel(logging.INFO)

def load_yaml_config(yaml_file):
    config = parse_config(yaml_file)
    return config


def run(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", dest="experiment")
    parser.add_argument("--loader", dest="loader")
    parser.add_argument("--db_config", dest="db_config")
    parser.add_argument("--dataset_config", dest="dataset_config")
    parser.add_argument("--benchmark_config", dest="benchmark_config")
    parser.add_argument(
        "--make_hdf5", nargs=4, metavar=("filetype", "inputfile", "hdf5file", "key")
    )
    parser.add_argument("--metrics", default=metrics.NOOP_METRICS, 
                        choices= [metrics.NOOP_METRICS, metrics.PANDAS_METRICS, 
                                 metrics.INFLUX_METRICS, metrics.GCP_METRICS],
                        dest="metrics")
    parser.add_argument("--report_only", dest="report_only")
    parser.add_argument("--report_file", dest="report_file")
    parser.add_argument("--report_folder", dest="report_folder")
    
    known_args, _ = parser.parse_known_args(argv)

    vecbench_ray = os.getenv("VECBENCH_RAY", "False")

    make_hdf5 = known_args.make_hdf5
    experiment = known_args.experiment

    if make_hdf5 is not None:
        make_hdf5_file(make_hdf5[0], make_hdf5[1], make_hdf5[2], make_hdf5[3])
        return

    if experiment is not None:
        experiment_config = load_yaml_config(known_args.experiment)
        experiment_config['metrics']=known_args.metrics
        experiment_config['report_folder']=known_args.report_folder
        experiment_config['report_file']=known_args.report_file
        experiment = Experiment(experiment_config)
        if "True" in vecbench_ray or 'MPLoader' in experiment_config['loaders'] :
            exec_steps = experiment.resolve()
            experiment.execute(exec_steps)
        return

    db_config = load_yaml_config(known_args.db_config)
    dataset_config = load_yaml_config(known_args.dataset_config)
    benchmark_config = load_yaml_config(known_args.benchmark_config)
    benchmark_config['config']['metrics']=known_args.metrics

    report_only = known_args.report_only
    report_file = known_args.report_file
    report_folder = known_args.report_folder
    if (report_only is not None and "Yes" in report_only):
        print("Report Only")
        generate_report(db_config, dataset_config, benchmark_config, report_file, report_folder)
        return

    loader = Loader(db_config, dataset_config, benchmark_config)
    if "MPLoader" in known_args.loader:
        logging.info(f"Loading in MPLoader.")
        loader.load_in_mploader()
    elif "RAYLoader" in known_args.loader:
        logging.info(f"Loading in RAYLoader.")
        loader.load_in_rayloader()


    if "MPLoader" in known_args.loader or "True" in vecbench_ray:
        # Generate the report for this run.
        generate_report(db_config, dataset_config, benchmark_config, report_file, report_folder)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
