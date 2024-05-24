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

from experiments.execute import Execution
from mp.raysubmitter import RayLoader
import sys
import os
import json

class Experiment:
    def __init__(self, config):
        self.config = config
        loader = self.config["loaders"]
        vecbench_ray = os.getenv("VECBENCH_RAY", "False")
        if "False" in vecbench_ray and "RAYLoader" in loader:
            rayloader = RayLoader()
            rayloader.ray_submit(sys.argv)
            return

    def resolve(self):
        print(f"Experiment:{self.config['name']}")
        print(f"Description:{self.config['description']}")
        print(f"Metrics:{self.config['metrics']}")
        print(f"Loaders:{self.config['loaders']}")

        exec_steps = []
        for benchmark_name in self.config["benchmarks"]:
            print(f"Benchmark name:{benchmark_name}")
            benchmark_configs = self.config["benchmarks"][benchmark_name]["configs"]
            datasets = self.config["benchmarks"][benchmark_name]["datasets"]
            benchmark_overrides = self.config["benchmarks"][benchmark_name]["overrides"]

            index_overrides = benchmark_overrides['index_config']
            if 'probes' in benchmark_overrides.keys():
                attr_name = 'probes'
                attr_overrides = benchmark_overrides['probes']
            elif 'num_leaves_to_search' in benchmark_overrides.keys():
                attr_overrides = benchmark_overrides['num_leaves_to_search']
                attr_name = 'num_leaves_to_search'

            benchmark_overrides = self.config["benchmarks"][benchmark_name]["overrides"]
            for index_override in index_overrides:
                print(f"index_override: {index_override}")
                benchmark_overrides['index_config'] = index_override
                for attr_override in attr_overrides:
                    benchmark_overrides[attr_name] = attr_override
                    # Create steps from experiment config
                    for benchmark_config in benchmark_configs:
                        for dataset in datasets:
                            for store in self.config["stores"]:
                                for loader in self.config["loaders"]:
                                    step = {
                                        "benchmark": benchmark_config,
                                        "overrides": dict(benchmark_overrides),
                                        "dataset": dataset,
                                        "store": store,
                                        "loader": loader,
                                        "metrics": self.config["metrics"],
                                        "report_folder": self.config["report_folder"],
                                        "report_file": self.config["report_file"],
                                    }
                                    exec_steps.append(step)
        return exec_steps

    def execute(self, exec_steps):
        for step in exec_steps:
            print(f"Executing: {json.dumps(step, sort_keys=True, indent=4)}")
            execution = Execution(step=step)
            execution.load_config()
            execution.execute()
