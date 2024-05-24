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
from workloads.workload import Workload
import logging
import metrics
from db.dbglobal import DBGlobal
from db.dbsetup import DBSetup

logging.getLogger().setLevel(logging.INFO)

class BasicBatchAnnWorkload(Workload):
    def __init__(self, db_config, config, table_name, dataset, gt_datasets, coordinator):
        super().__init__(db_config, config, table_name, dataset, gt_datasets, coordinator)
        self.retrieved_ids = []
        self.run_id = config['run_id']
        self.db_config = db_config
        self.searchdata = dataset
        self.dimensions = len(self.searchdata[0])
        self.search_limit = int(config["search_limit"])
        self.config = config
        self.algo = config["algo"]
        self.probes = 0
        self.index_type = config["index_type"]
        self.batch_size = config["batch_size"]
        self.num_leaves_to_search = 0
        if any(index in self.index_type for index in ["scann", "TREE_AH", "TREE_SQ"]):
            self.num_leaves_to_search = int(config["num_leaves_to_search"])
        else:
            self.probes = int(config["probes"])
        self.table_name = table_name
        self.duration = int(config["duration_in_seconds"])

    def load(self, worker_number):
        pid = os.getpid()
        self.metrics = metrics.get_metrics(self.config["metrics"], self.run_id)
        self.db = DBSetup(self.db_config)
        self.db.load_table(self.table_name)
        search_algo = DBGlobal.algo_to_pred(self.algo)
        tags = {
            "tool": "BasicAnnWorkload",
            "worker": str(pid),
            "type": self.db.type,
            "algo": self.algo,
            "worker_number": str(worker_number),
            "probes": str(self.probes),
            "index_type": str(self.index_type),
            "dimensions": str(self.dimensions),
        }
        num_entries_processed = 0
        logging.info(
            f"Starting load worker:{pid} worker_number {worker_number} Searching: {len(self.searchdata)}"
        )

        self.db.configure_search_session(self.config)

        while self.run:
            for i in range(0, len(self.searchdata), self.batch_size):
                batch = self.searchdata[i:i+self.batch_size]
                start = time.time()
                ##
                # We expect the store to return a list of list of tuples:
                # [[(97478,), (262700,), (846101,), (671078,), (232287,)...]...]
                ##
                resp = self.db.annbatchsearch(batch, self.search_limit, search_algo)
                end = time.time()
                returned_rows = self.db.returned_rows_batch(resp)

                assert len(returned_rows) == len(batch)
                num_entries_processed += 1
                self.metrics.collect("annsearch", tags, "elapsed", (end - start))
                self.metrics.collect(
                    "annsearch", tags, "searchcount", num_entries_processed
                )
                for j in range(len(batch)):
                    self.retrieved_ids.append({'truth_id': i+j, 'search_vector': batch[j], 'returned_ids': returned_rows[j]})

                if self.run == False:
                    break
            if self.duration == 0:
                break
        self.complete_phase_and_wait(worker_number)
        self.process_recall(tags)
        self.metrics.close()
