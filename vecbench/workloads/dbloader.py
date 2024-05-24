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

import logging
import time
import os
import metrics
from db.dbsetup import DBSetup
from workloads.workload import Workload

logging.getLogger().setLevel(logging.INFO)


class DBLoader(Workload):
    def __init__(self, db_config, config, table_name, dataset, start, end):
        self.db_config = db_config
        self.config = config
        self.run_id = config['run_id']
        self.table_name = table_name
        self.distance_metric = config["algo"]
        self.dataset = dataset
        self.length = len(self.dataset)
        self.dimensions = len(self.dataset[0])
        self.start = start
        self.end = end

    def load(self, worker_number):
        pid = os.getpid()
        self.metrics = metrics.get_metrics(metrics.NOOP_METRICS, self.run_id)
        db = DBSetup(self.db_config)
        tags = {
            "tool": "DBLoader",
            "worker": str(pid),
            "type": self.db_config['type'],
            "worker_number": str(worker_number),
            "dimensions": str(self.dimensions),
            "length": str(self.length),
            "start": str(self.start),
            "end": str(self.end),
        }
        logging.info(f"Starting dbloader worker:{pid} worker_number {worker_number} Inserting: {len(self.dataset)}")
        start = time.time()
        db.load_dataset(self.table_name, self.dataset, self.start, self.end, self.distance_metric)
        end = time.time()
        self.metrics.collect("dbloader", tags, "elapsed", (end - start))
        self.metrics.close()
