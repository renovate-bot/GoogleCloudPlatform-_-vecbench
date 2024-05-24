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

import multiprocessing as mp
from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler

try:
   mp.set_start_method('fork', force=True)
except RuntimeError:
   pass

class TimedWorker(Thread):
    def __init__(self, workload, benchmark_config):
        Thread.__init__(self)
        config = benchmark_config['config']
        numworkers = int(config['number_of_workers'])
        duration = int(config['duration_in_seconds'])

        self.loader = MPLoader()
        self.loader.run_single(workload, numworkers)
        if duration > 0:
            scheduler = BackgroundScheduler()
            scheduler.add_job(self.cancel, 'interval', seconds=duration)
            scheduler.start()
        self.start()

    def cancel(self):
        self.loader.stop_load()

    def run(self):
        self.loader.start_load()

class MPLoader():
    def run_single(self, workload, num_workers):
        self.processes = [mp.Process(target=workload.load, args=(x,)) for x in range(num_workers)]

    def run_array(self, workloads, num_workers):
        self.processes = [mp.Process(target=workloads[x].load, args=(x,)) for x in range(num_workers)]

    def start_load(self):
        for p in self.processes:
            p.start()

        # Exit the completed processes
        for p in self.processes:
            p.join()

    def stop_load(self):
        for p in self.processes:
            p.terminate()
