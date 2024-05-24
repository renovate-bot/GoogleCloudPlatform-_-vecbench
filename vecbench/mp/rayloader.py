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

import ray
import time
import asyncio

from db.dbsetup import DBSetup
import numpy as np
from mp.coordinator import Coordinator

@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.ready_event = asyncio.Event()
        self.waiters = 0

    def get_waiters(self):
        return self.waiters

    def send(self, clear=False):
        self.ready_event.set()
        if clear:
            self.ready_event.clear()

    async def wait(self, should_wait=True):
        self.waiters = self.waiters + 1
        if should_wait:
            await self.ready_event.wait()


@ray.remote(scheduling_strategy="SPREAD", resources={"vecsearch": 1})
def do_load(worker_number, db_config, table_name, dataset, start, end, distance_metric):
    print(f"launched Loader {worker_number}! start:{start} end: {end}")
    db = DBSetup(db_config)
    try:
        db.load_dataset(table_name, dataset, start, end, distance_metric)
    except ray.exceptions.TaskCancelledError as E:
        print("Task finished")


@ray.remote(scheduling_strategy="SPREAD", resources={"vecsearch": 1})
def do_loop(
    worker_number, dyna_workload, db_config, config, table_name, dataset, signal, ground_truth_datasets, coordinator
):
    ray.get(signal.wait.remote())
    print("launched!")
    try:
        workload = dyna_workload(db_config, config, table_name, dataset, ground_truth_datasets, coordinator)
        workload.load(worker_number)
    except (KeyboardInterrupt) as e:
        print("Task finished")

    data = ""
    csvfile=f"downloads/db_{config['run_id']}_{os.getpid()}.csv"
    if os.path.isfile(csvfile):
        with open(csvfile) as f:
            data = f.read()
    return ray.put(data)


def init_ray():
    ray.init()
    print(
        """This cluster consists of
            {} nodes in total
            {} CPU resources in total
        """.format(
            len(ray.nodes()), ray.cluster_resources()["CPU"]
        )
    )


def run_dbload_in_ray(db_config, table_name, db_dataset, num_loaders, start, distance_metric):
    split_dataset = np.array_split(db_dataset, num_loaders)
    split_dataset_ref = ray.put(split_dataset)
    load_object_refs = []

    for worker_number in range(num_loaders):
        split_dataset_ref = ray.put(split_dataset[worker_number])
        end = start + len(split_dataset[worker_number])
        load_object_refs.append(
            do_load.remote(worker_number, db_config, table_name, split_dataset_ref, start, end, distance_metric)
        )
        start = end

    ready_refs, _ = ray.wait(load_object_refs, num_returns=num_loaders, timeout=None)
    del load_object_refs
    del split_dataset_ref
    return start


def run_in_ray_workload(
    db_config, benchmark_config, dyna_workload, table_name, search_dataset, ground_truth_datasets
):
    config = benchmark_config["config"]
    numworkers = int(config["number_of_workers"])
    duration = int(config["duration_in_seconds"])
    dataset = ray.put(search_dataset)
    signal = SignalActor.options(max_concurrency=numworkers*2).remote()
    coordinator = Coordinator(numworkers, "RAYLoader")
    object_refs = [
        do_loop.remote(
            worker_number, dyna_workload, db_config, config, table_name, dataset, signal, ground_truth_datasets, coordinator)
        for worker_number in range(numworkers)
    ]

    ready_workloads = 0
    while ready_workloads < numworkers:
        print(f"ready_workloads: {ready_workloads}")
        ready_workloads = ray.get(signal.get_waiters.remote())
        time.sleep(1)
    print(f"launching {ready_workloads} workloads.")

    # Kick off the run
    ray.get(signal.send.remote())
    start = time.time()
    # If duration is zero, we let the workload run to completion
    if duration > 0:
        time.sleep(duration)
        print("Duration expired, cancelling tasks.")
        for object_ref in object_refs:
            ray.cancel(object_ref)

    ready_refs, remaining_refs = ray.wait(
        object_refs, num_returns=numworkers, timeout=None
    )

    end = time.time()
    datas = ray.get(ready_refs)
    print(f"Completed {len(datas)} jobs in {(end - start)} seconds.")
    # Persist metrics
    i = 0
    for bufref in datas:
        d = ray.get(bufref)
        file_object = open(f"downloads/db_{config['run_id']}_{i}.csv", "w")
        file_object.write(d)
        file_object.close()
        i = i + 1
