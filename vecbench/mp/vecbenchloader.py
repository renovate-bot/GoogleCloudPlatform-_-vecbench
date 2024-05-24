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

from workloads.benchmark import BenchmarkSetup
from workloads.dbloader import DBLoader
from mp.coordinator import Coordinator
from mp.mploader import TimedWorker, MPLoader
from mp.raysubmitter import RayLoader
import sys
import os
from mp.rayloader import init_ray, run_dbload_in_ray, run_in_ray_workload
import numpy as np
from report.report import generate_report
import ray
import uuid

class Loader:
    def __init__(self, db_config, dataset_config, benchmark_config):
        self.db_config = db_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self.benchmarksetup = BenchmarkSetup(
            db_config, dataset_config, benchmark_config
        )
        self.table_name = dataset_config["type"]
        self.db_dataset_key = dataset_config["config"]["db_dataset_key"]
        self.db_recreate = dataset_config["config"]["db_recreate"]
        self.number_loaders = int(dataset_config["config"]["number_loaders"])
        self.distance_metric = benchmark_config["config"]["algo"]
        self.search_key = benchmark_config["config"]["search_key"]
        if "queries_num" in benchmark_config["config"].keys():
            self.queries_num = benchmark_config["config"]["queries_num"]
        else:
            self.queries_num = None
        if "ground_truth_keys" in benchmark_config["config"].keys():
            self.gt_keys = benchmark_config["config"]["ground_truth_keys"]
        else:
            self.gt_keys = None
        if "report_template" in benchmark_config["config"].keys():
            self.report_template = benchmark_config["config"]["report_template"]
        else:
            self.report_template = None

    def setup_io(self):
        dataset_io = self.benchmarksetup.setup_datasets_io()
        database_io = self.benchmarksetup.setup_db_io()
        return dataset_io, database_io

    def setup_schema(self):
        dataset_io, database_io = self.setup_io()
        db_dataset_files = dataset_io.get_db_dataset_files()

        # Inspect the first file to create the schema
        dbdataset = dataset_io.load_dataset_file(db_dataset_files[0])
        dimension = len(dbdataset[self.db_dataset_key][0])
        database_io.create_table(self.table_name, self.db_recreate, dimension, self.benchmark_config)
        return db_dataset_files

    def load_in_mploader(self):
        # Generate a random run id
        run_id = uuid.uuid4()
        self.benchmark_config["config"]["run_id"] = run_id
        self.db_config["config"]["run_id"] = run_id
        self.db_config["config"]["metrics"] = self.benchmark_config["config"]["metrics"]
        # Setup IO
        dataset_io, database_io = self.setup_io()
        if self.db_recreate:
            # Create the db schema
            db_dataset_files = self.setup_schema()

            # Iterate over db dataset files and load them into the table.
            start = 0
            for db_dataset_file in db_dataset_files:
                dbdataset = dataset_io.load_dataset_file(db_dataset_file)[
                    self.db_dataset_key
                ]
                split_dataset = np.array_split(dbdataset, self.number_loaders)

                loaders = [] 
                for x in range(self.number_loaders):
                    end = start + len(split_dataset[x])
                    dbloader = DBLoader(self.db_config, self.benchmark_config["config"], self.table_name, split_dataset[x], start, end) 
                    loaders.append(dbloader)
                    start = end

                mploader = MPLoader()
                mploader.run_array(loaders, self.number_loaders)
                mploader.start_load()
                # Remove the dataset file to lower memory utilization.
                dataset_io.unload_dataset_file(db_dataset_file)

            # Force Index creation since we just recreated the table
            self.benchmark_config["config"]["index_recreate"] = True

        database_io.load_table(self.table_name, self.distance_metric)
        database_io.set_value(self.table_name)
        self.benchmarksetup.index_dataset(self.benchmark_config)


        # Load the search dataset
        search_dataset_file = dataset_io.get_search_dataset_files()

        search_dataset = dataset_io.load_dataset_file(search_dataset_file)[self.search_key]
        if self.queries_num is not None:
            search_dataset = search_dataset[:self.queries_num]

        # Load the ground truth datasets
        ground_truth_datasets = [] 
        if self.gt_keys:
            ground_truth_dataset_files = dataset_io.get_ground_truth_dataset_files()
            for i, gt_dataset_file in enumerate(ground_truth_dataset_files):
                dataset = dataset_io.load_dataset_file(gt_dataset_file)[self.gt_keys[i]]
                if self.queries_num is not None:
                    dataset = dataset[:self.queries_num]
                ground_truth_datasets.append(dataset)
            # Ensure that key is provided per ground truth dataset
            assert(len(self.gt_keys) == len(ground_truth_datasets))
        
        # Load workload class
        dyna_workload = self.benchmarksetup.load_benchmark()
        config = self.benchmark_config["config"]
        self.coordinator = Coordinator(int(config['number_of_workers']), "MPLoader")
        workload = dyna_workload(self.benchmarksetup.db_config, config, self.table_name, search_dataset, ground_truth_datasets, self.coordinator)
        # Schedule the workload class
        timedWorker = TimedWorker(workload, self.benchmark_config)
        timedWorker.join()

    def load_in_rayloader(self):
        vecbench_ray = os.getenv("VECBENCH_RAY", "False")
        # First leg submits the job
        if "False" in vecbench_ray:
            rayloader = RayLoader()
            # rayloader.ray_launch()
            rayloader.ray_submit(sys.argv)
            # rayloader.ray_down()
            return
        # Second leg runs in the driver and launches remote tasks.
        # Init Ray
        init_ray()
        # Generate a random run id
        run_id = uuid.uuid4()
        self.benchmark_config["config"]["run_id"] = run_id
        self.db_config["config"]["run_id"] = run_id
        self.db_config["config"]["metrics"] = self.benchmark_config["config"]["metrics"]

        dataset_io, database_io = self.setup_io()
        if self.db_recreate:
            db_dataset_files = self.setup_schema()
            start = 0
            for db_dataset_file in db_dataset_files:
                dbdataset = dataset_io.load_dataset_file(db_dataset_file)[self.db_dataset_key]
                start = run_dbload_in_ray(self.db_config, self.table_name, dbdataset, self.number_loaders, start, self.distance_metric)
                dataset_io.remove_dataset_file(db_dataset_file)

            # Force Index creation since we just recreated the table
            self.benchmark_config["config"]["index_recreate"] = True

        database_io.load_table(self.table_name, self.distance_metric)
        database_io.set_value(self.table_name)
        self.benchmarksetup.index_dataset(self.benchmark_config)


        # Load the search dataset
        search_dataset_file = dataset_io.get_search_dataset_files()
        search_dataset = dataset_io.load_dataset_file(search_dataset_file)[
            self.search_key
        ]
        
        # Load the ground truth datasets
        ground_truth_datasets = [] 
        if self.gt_keys:
            ground_truth_dataset_files = dataset_io.get_ground_truth_dataset_files()
            for i, gt_dataset_file in enumerate(ground_truth_dataset_files):
                ground_truth_datasets.append(dataset_io.load_dataset_file(gt_dataset_file)[self.gt_keys[i]])
            # Ensure that key is provided per ground truth dataset
            assert(len(self.gt_keys) == len(ground_truth_datasets))
        
        # Load workload class
        dyna_workload = self.benchmarksetup.load_benchmark()
        run_in_ray_workload(
            self.db_config, self.benchmark_config, dyna_workload, self.table_name, search_dataset, ground_truth_datasets)

        ray.shutdown()
