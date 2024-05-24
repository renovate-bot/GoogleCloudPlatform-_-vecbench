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
import random
import metrics
import numpy as np
import itertools
import signal, os
from time import sleep
import math

logging.getLogger().setLevel(logging.INFO)

def norm(a):
    return np.sum(a**2) ** 0.5

def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)

def euclidean(a, b):
    return norm(a - b)

def knn_threshold(data, count, epsilon, algo_type = None):
    if algo_type == "vector_ip_ops":
        return data[count - 1] - epsilon
    return data[count - 1] + epsilon

def ip(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

recall_metrics = {
        "hamming": {
            "distance": lambda a, b: np.sum(a.astype(np.bool_) ^ b.astype(np.bool_)),
            "distance_valid": lambda a: True,
        },
        # return 1 - jaccard similarity, because smaller distances are better.
        "jaccard": {
            "distance": lambda a, b: 1 - jaccard(a, b),
            "distance_valid": lambda a: a < 1 - 1e-5,
        },
        # vector_l2_ops
        "vector_l2_ops": {
            "distance": lambda a, b: euclidean(a, b),
            "distance_valid": lambda a: True,
        },
        # vector_cosine_ops
        "vector_cosine_ops": {
            "distance": lambda a, b: 1 - np.dot(a, b) / (norm(a) * norm(b)),
            "distance_valid": lambda a: True,
        },
        "vector_ip_ops": {
            "distance": lambda a,b: ip(a, b),
            "distance_valid": lambda a: True,
        },
    }

class Workload:
    def __init__(self, db, config, table_name, dataset, gt_datasets, coordinator):
        self.db = db
        self.run = True
        self.config = config
        self.run_id = config['run_id']
        self.dataset = dataset
        self.table_name = table_name
        self.sleep_time = 0.01
        self.coordinator = coordinator
        # Set the signal handler
        signal.signal(signal.SIGTERM, self.handler)

        if "ground_truth_keys" in config.keys():
            if len(config["ground_truth_keys"]) == 1 and ("distances" in config["ground_truth_keys"] or "neighbors" in config["ground_truth_keys"]):
                self.recall_type = config["ground_truth_keys"][0]
                if (config["ground_truth_keys"][0] == 'distances'):
                    self.distance_gt = gt_datasets[0]
                    self.neighbors_gt = None
                elif (config["ground_truth_keys"][0] == 'neighbors'):
                    self.neighbors_gt = gt_datasets[0]
                    self.distance_gt = None
            elif len(config["ground_truth_keys"]) == 2 and "distances" in config["ground_truth_keys"] and "neighbors" in config["ground_truth_keys"]:
                self.recall_type = 'ALL'
                if (config["ground_truth_keys"][0] == 'distances'):
                    self.distance_gt = gt_datasets[0]
                    self.neighbors_gt = gt_datasets[1]
                elif (config["ground_truth_keys"][0] == 'neighbors'):
                    self.neighbors_gt = gt_datasets[0]
                    self.distance_gt = gt_datasets[1]
        else:
            self.recall_type = None

    def complete_phase_and_wait(self, num_worker):
        logging.info(f"Worker: {num_worker} completed phase.")
        self.coordinator.block_and_wait()
        logging.info(f"Worker: {num_worker} finishing.")

    def load(self, worker_number):
        pid = os.getpid()
        self.metrics = metrics.get_metrics(self.config["metrics"], self.run_id)
        tags = {'tool':'calibrate', 'worker':str(pid), 'worker_number':str(worker_number)}
        logging.info(f"Starting load worker:{pid} worker_number {worker_number} Searching: {len(self.dataset)}")
        while self.run:
            start = time.time()
            time.sleep(self.sleep_time)
            end = time.time()
            self.metrics.collect("calibrate", tags, "elapsed", (end - start))
        self.metrics.close()

    def handler(self, signum, frame):
        self.run = False

    # Generate a new random embedding as per the given embedding format
    def generate_embedding(self, basedatum):
        minvalue = min(basedatum)
        maxvalue = max(basedatum)
        newdatum = []
        for _ in range(self.ndim):
            newdatum.append(random.uniform(minvalue, maxvalue))
        return newdatum

    def calculate_recall_based_on_distances_only(self, algo_type, searchdatum, distance_dataset, query_result, tags):
        eps = 1e-3
        actual = 0
        threshold = knn_threshold(distance_dataset, self.search_limit, eps, algo_type)
        for result in query_result:
            single_result = result[0]
            single_result_datum = single_result[1]
            if isinstance(searchdatum, list):
                searchdatum = np.array(searchdatum)
            if isinstance(single_result[1], list):
                single_result_datum = np.array(single_result[1])
            distance = recall_metrics[algo_type]["distance"](searchdatum, single_result_datum)
            if algo_type == 'vector_ip_ops':
                if distance >= threshold:
                    actual += 1
            else:
                if distance <= threshold:
                    actual += 1

        self.metrics.collect("annsearch", tags, "recall_d", actual / self.search_limit)
    
    
    def calculate_recall_based_on_ids_only(self, distance_dataset, query_result, tags):
        match_count = 0
        for result in query_result:
            id = result[0]
            if (id[0]) in distance_dataset[0:self.search_limit]:
                match_count += 1
        self.metrics.collect("annsearch", tags, "recall_n", match_count / self.search_limit)

    def calculate_recall_without_distance_ties(self, true_ids, query_result, tags):
        run_ids = []
        for id in query_result:
            run_ids.append(id[0])
        recall = len(set(true_ids[:self.search_limit]) & set(run_ids))
        self.metrics.collect("annsearch", tags, "recall_wo_ties", recall / self.search_limit)
    
    def calculate_recall_based_on_ids_and_distances(self, true_dists, true_ids, query_result, tags):

      found_tie = False
      gt_size = np.shape(true_dists)[0]
      run_ids = []
      for result in query_result:
        if len(result) > 0:
            id = result[0]
            run_ids.append(id[0])

      if gt_size==self.search_limit:
          recall =  len(set(true_ids[:self.search_limit]) & set(run_ids))
      else:
          dist_tie_check = true_dists[self.search_limit-1] # tie check anchored at count-1 in GT dists
     
          set_end = gt_size

          for i in range(self.search_limit, gt_size):
            is_close = abs(dist_tie_check - true_dists[i]) < 1e-6 
            if not is_close:
              set_end = i
              break

          found_tie = set_end > self.search_limit

          recall =  len(set(true_ids[:set_end]) & set(run_ids))
      self.metrics.collect("annsearch", tags, "recall_d_n", recall / self.search_limit)

    def process_recall(self, tags):
        for retrieved_id in self.retrieved_ids:
            truth_id = retrieved_id['truth_id']
            search_vector = retrieved_id['search_vector']
            returned_vectors = []
            ids = list(itertools.chain.from_iterable(retrieved_id['returned_ids']))
            ##
            # get_by_id_batch is expected to return a list of (vector_id, vector) eg:
            # [[(97478, array([-0.013667, -0.33188 ,  0.41867 ,  0.044317, -0.41382 , -0.43985 ,], dtype=float32))],...]
            ##
            returned_vectors = self.db.get_by_id_batch(ids)
            self.calc_all_recalls(truth_id, search_vector, returned_vectors, tags)

    def calc_all_recalls(self, i, searchdatum, returned_rows, tags):
        if self.recall_type == 'neighbors':
            self.calculate_recall_based_on_ids_only(self.neighbors_gt[i], returned_rows, tags)
        elif self.recall_type == 'distances':
            self.calculate_recall_based_on_distances_only(self.algo, searchdatum, self.distance_gt[i], returned_rows, tags)
        elif self.recall_type == 'ALL':
            self.calculate_recall_based_on_distances_only(self.algo, searchdatum, self.distance_gt[i], returned_rows, tags)
            self.calculate_recall_based_on_ids_only(self.neighbors_gt[i], returned_rows, tags)
            self.calculate_recall_based_on_ids_and_distances(self.distance_gt[i], self.neighbors_gt[i], returned_rows, tags)
    def process_recall_mixed(self, tags):
        for i, step in enumerate(self.retrieved_ids):
            for retrieved_id in self.retrieved_ids[step]:
                truth_id = retrieved_id['truth_id']
                search_vector = retrieved_id['search_vector']
                returned_vectors = []
                ids = list(itertools.chain.from_iterable(retrieved_id['returned_ids']))
                returned_vectors = self.db.get_by_id_batch(ids)
                self.calc_all_recalls_mixed(truth_id, search_vector, returned_vectors, tags, i)

    def calc_all_recalls_mixed(self, i, searchdatum, returned_rows, tags, search_phase):
        if self.recall_type == 'neighbors':
            self.calculate_recall_based_on_ids_only(self.neighbors_gt[search_phase][i], returned_rows, tags)
        elif self.recall_type == 'distances':
            self.calculate_recall_based_on_distances_only(self.algo, searchdatum, self.distance_gt[search_phase][i], returned_rows, tags)
        elif self.recall_type == 'ALL':
            self.calculate_recall_based_on_distances_only(self.algo, searchdatum, self.distance_gt[search_phase][i], returned_rows, tags)
            self.calculate_recall_based_on_ids_only(self.neighbors_gt[search_phase][i], returned_rows, tags)
            self.calculate_recall_based_on_ids_and_distances(self.distance_gt[search_phase][i], self.neighbors_gt[search_phase][i], returned_rows, tags)

    def calculate_distances(self, true_n_emb, searchdatum, algo_type):
        return recall_metrics[algo_type]["distance"](searchdatum, true_n_emb)
