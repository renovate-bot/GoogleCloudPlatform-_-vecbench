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

import deepdish as dd
import logging
from google.cloud import storage
import os
import struct
import numpy as np

loaded_datasets = {}

class DatasetIOSetup:
    def __init__(self, dataset_config, benchmark_config):
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config

    def parse_dataset_file(self, dataset_file):
        s = dataset_file.split("/")
        bucket = s[2]
        source_blob = "/".join(s[3:])
        file_name = s[-1:][0]
        destination_file = f"downloads/{file_name}"
        return bucket, source_blob, file_name, destination_file

    def download_blob(self, dataset_file):
        logging.info(f"Downloading file:{dataset_file}")
        bucket, source_blob, _, destination_file = self.parse_dataset_file(dataset_file)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(destination_file)
        logging.info(f"Downloading file:{dataset_file} complete!")

    def get_db_dataset_files(self):
        dataset_files = self.dataset_config["config"]["dataset_files"]
        dataset_list = dataset_files.split(", ")
        return dataset_list

    def get_search_dataset_files(self):
        dataset_files = self.benchmark_config["config"]["search_dataset"]
        return dataset_files

    def get_ground_truth_dataset_files(self):
        dataset_files = self.benchmark_config["config"]["ground_truth_datasets"]
        return dataset_files

    def load_dataset_file(self, dataset_file):
        if dataset_file in loaded_datasets.keys():
            return loaded_datasets[dataset_file]

        logging.info(f"Loading {dataset_file}")
        _, _, file_name, destination_file = self.parse_dataset_file(dataset_file)
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
        if not os.path.isfile(destination_file):
            self.download_blob(dataset_file)

        hdf5_file = destination_file
        if hdf5_file not in loaded_datasets.keys():
            dataset = self.analyze(hdf5_file)
        loaded_datasets[dataset_file] = dataset
        return dataset

    def unload_dataset_file(self, dataset_file):
        if dataset_file in loaded_datasets.keys():
            logging.info(f"Unloading {dataset_file}")
            del loaded_datasets[dataset_file]

    def remove_dataset_file(self, dataset_file):
        if dataset_file in loaded_datasets.keys():
            del loaded_datasets[dataset_file]
        _, _, file_name, destination_file = self.parse_dataset_file(dataset_file)
        os.remove(destination_file)

    def analyze(self, hdf5_file):
        dataset = dd.io.load(hdf5_file)
        logging.info("----------------------------------------------")
        logging.info(f"File:{hdf5_file}")
        for key in dataset.keys():
            logging.info(
                f"Key:{key} Dimensions:{len(dataset[key][0])} Size:{dataset[key].size}"
            )
        logging.info("----------------------------------------------")
        return dataset

