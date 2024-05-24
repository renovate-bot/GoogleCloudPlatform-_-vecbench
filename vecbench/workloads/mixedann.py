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
import numpy as np
from collections import defaultdict
from db.dbglobal import DBGlobal
from db.dbsetup import DBSetup
from pyaml_env import parse_config
from datasets.dataset import DatasetIOSetup
logging.getLogger().setLevel(logging.INFO)


class MixedAnnWorkload(Workload):
    def __init__(self, db_config, config, table_name, dataset, gt_datasets, coordinator):
        super().__init__(db_config, config, table_name, dataset, gt_datasets, coordinator)
        self.retrieved_ids = {}
        self.run_id = config['run_id']
        self.db_config = db_config
        self.searchdata = dataset
        self.config = config
        self.insert_file_setup = DatasetIOSetup(self.db_config, {'config':self.config})
        self.search_limit = int(config["search_limit"])
        self.algo = config["algo"]
        self.probes = 0
        self.index_type = config["index_type"]
        self.runbook_config = parse_config(config['runbook'])
        self.insert_file = self.runbook_config['Insert_File']['files']
        self.insertdata = self.load_insert_file(self.insert_file)
        self.num_workers = int(config['number_of_workers'])
        if self.num_workers != 1:
            logging.error("Mixed Workloads can be run with only 1 worker.")
            exit(1)
        self.num_leaves_to_search = 0
        if any(index in self.index_type for index in ["scann", "TREE_AH", "TREE_SQ"]):
            self.num_leaves_to_search = int(config["num_leaves_to_search"])
        else:
            sef.probes = int(config["probes"])

        self.table_name = table_name
        self.ndim = len(dataset[0])


    def load_insert_file(self, filenames):
        datasets = []
        for filename in filenames:
            datasets.append(self.insert_file_setup.load_dataset_file(filename)[self.runbook_config['Insert_File']['key']])
        final_dataset = np.concatenate(datasets, axis=0)
        return final_dataset

    def load(self, worker_number):
        pid = os.getpid()
        self.metrics = metrics.get_metrics(self.config["metrics"], self.run_id)
        self.db = DBSetup(self.db_config)
        self.db.load_table(self.table_name, self.algo)
        search_algo = DBGlobal.algo_to_pred(self.algo)
        datasetsize = self.db.anndatasetsize(self.table_name)
        print(f"number of rows in the dataset {datasetsize}")
        tags = {
            "tool": "MixedAnnWorkload",
            "worker": str(pid),
            "type": self.db.type,
            "algo": self.algo,
            "worker_number": str(worker_number),
            "Dataset size": str(datasetsize),
            "dimensions": str(self.ndim),
        }
        read_processed = 0
        insert_processed = 0
        delete_processed = 0
        logging.info(f"Starting load worker:{pid} worker_number {worker_number}")

        self.db.configure_search_session(self.config)

        
        try:
            for operation in self.runbook_config['Operations']:
                oper = self.runbook_config['Operations'][operation]
                if oper['type']=="Search":
                    logging.info(f"Step {operation}: Searching {len(self.searchdata)} queries from table")
                    for i, searchdatum in enumerate(self.searchdata):
                        start = time.time()
                        resp = self.db.annsearch(
                            searchdatum, self.search_limit, search_algo
                        )
                        returned_ids = self.db.returned_rows(resp)
                        end = time.time()
                        assert len(returned_ids) > 0
                        read_processed += 1
                        self.metrics.collect(
                            "mixedann", tags, "read_elapsed", (end - start)
                        )
                        self.metrics.collect(
                            "mixedann", tags, "readoperationcount", read_processed
                        )

                        if 'step_'+str(operation) not in self.retrieved_ids:
                            self.retrieved_ids['step_'+str(operation)] =  [{'truth_id': i, 'search_vector': searchdatum, 'returned_ids':returned_ids}]
                        else:
                            self.retrieved_ids['step_'+str(operation)].append({'truth_id': i, 'search_vector': searchdatum, 'returned_ids':returned_ids})
                elif oper['type']=="Insert":
                    logging.info(f"Step {operation}: Inserting {oper['end'] - oper['start']} rows into table")
                    for insert_id in range(oper['start'], oper['end']):
                        insertdatum = [x for x in self.insertdata[self.runbook_config['Insert_File']['start_id']-insert_id]]
                        start = time.time()
                        ret = self.db.anninsert(insertdatum, self.table_name, insert_id=insert_id)
                        insert_processed += 1
                        end = time.time()
                        self.metrics.collect(
                            "mixedann", tags, "insert_elapsed", (end - start)
                        )
                        self.metrics.collect(
                            "mixedann", tags, "insertoperationcount", insert_processed
                        )
                elif oper['type']=="Delete":
                    logging.info(f"Step {operation}: Deleting {oper['end'] - oper['start']} rows from table")
                    for delete_id in range(oper['start'], oper['end']):
                        start = time.time()
                        ret = self.db.anndelete(delete_id, self.table_name)
                        if ret.rowcount == 0:
                            # if matching row not found to delete, skip tracking it.
                            continue
                        delete_processed += 1
                        end = time.time()
                        self.metrics.collect(
                            "mixedann", tags, "delete_elapsed", (end - start)
                        )
                        self.metrics.collect(
                            "mixedann", tags, "deleteoperationcount", delete_processed
                        )
                else:
                    logging.error(f"unexpected operation weight: {oper}")
        except IOError as e:
            logging.error(
                f"failed while processing queries in mixed workload {e.strerror}"
            )
        finally:
            self.complete_phase_and_wait(worker_number)
            self.process_recall_mixed(tags)
            self.metrics.close()