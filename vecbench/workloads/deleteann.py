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

import os
import time
import random
from workloads.workload import Workload
import logging
import metrics
from db.dbsetup import DBSetup

logging.getLogger().setLevel(logging.INFO)


class DeleteAnnWorkload(Workload):
    def __init__(self, db_config, config, table_name, dataset, gt_datasets, coordinator):
        super().__init__(db_config, config, table_name, dataset, gt_datasets, coordinator)
        self.db_config = db_config
        self.run_id = config['run_id']
        self.searchdata = dataset
        self.algo = config["algo"]
        self.table_name = table_name
        self.dimensions = len(self.searchdata[0])
        self.config = config
        self.duration = int(config["duration_in_seconds"])

    def load(self, worker_number):
        pid = os.getpid()
        self.metrics = metrics.get_metrics(self.config["metrics"], self.run_id)
        self.db = DBSetup(self.db_config)
        self.db.load_table(self.table_name, self.algo)
        datasetsize = self.db.anndatasetsize(self.table_name)
        tags = {
            "tool": "DeleteAnnWorkload",
            "worker": str(pid),
            "type": self.db.type,
            "Dataset size": str(datasetsize),
            "worker_number": str(worker_number),
            "dimensions": str(self.dimensions),
        }
        num_entries_processed = 0
        logging.info(
            f"Starting load worker:{pid} worker_number {worker_number} Dataset size: {datasetsize} Deleting..."
        )

        while self.run:
            # searchdata is not being used here but just referenced to iterate
            # same number of times as other workload
            for _ in self.searchdata:
                start = time.time()
                ret = self.db.anndelete(random.randint(1, datasetsize), self.table_name)
                if ret.rowcount == 0:
                    # if matching row not found to delete, skip tracking it.
                    continue
                end = time.time()
                num_entries_processed += 1
                self.metrics.collect("anndelete", tags, "elapsed", (end - start))
                self.metrics.collect(
                    "anndelete", tags, "deletecount", num_entries_processed
                )

                if self.run == False:
                    break
            if self.duration == 0:
                break 
        
        self.metrics.close()
               
