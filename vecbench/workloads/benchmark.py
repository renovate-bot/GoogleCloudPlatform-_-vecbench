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

import importlib
from datasets.dataset import DatasetIOSetup
from db.dbsetup import DBSetup


class BenchmarkSetup:
    def __init__(self, db_config, dataset_config, benchmark_config):
        self.db_config = db_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self.datasets_IO_setup = None
        self.db = None

    def setup_datasets_io(self):
        if self.datasets_IO_setup == None:
            self.datasets_IO_setup = DatasetIOSetup(self.dataset_config, self.benchmark_config)
        return self.datasets_IO_setup

    def setup_db_io(self):
        if self.db == None:
            self.db = DBSetup(self.db_config)
        return self.db

    def load_dataset(self):
        self.db.load_dataset(self.dataset_name, self.db_dataset, 1, len(self.db_dataset))
    
    def index_dataset(self, benchmark_config):
        self.db.index_dataset(
            benchmark_config
        )

    def load_benchmark(self):
        type = self.benchmark_config["type"]
        class_name = self.benchmark_config["class"]
        module = importlib.import_module(f"{type}")
        dyna_workload = getattr(module, f"{class_name}")
        return dyna_workload
