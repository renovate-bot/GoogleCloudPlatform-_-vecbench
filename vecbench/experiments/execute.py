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


from pyaml_env import parse_config
from mp.vecbenchloader import Loader
import logging
import os
from report.report import generate_report
import json

logging.getLogger().setLevel(logging.INFO)

def load_yaml_config(yaml_file):
    config = parse_config(yaml_file)
    return config

class Execution:
    def __init__(self, step):
      self.step = step
      
    def load_config(self):
      self.metrics = self.step['metrics']
      self.report_folder = self.step['report_folder']
      self.report_file = self.step['report_file']
      self.db_config = load_yaml_config(self.step['store'])
      self.dataset_config = load_yaml_config(self.step['dataset'])
      self.benchmark_config = load_yaml_config(self.step['benchmark'])
      self.benchmark_overrides = self.step['overrides']
      print(f"Overrides: {self.benchmark_overrides}")
      for override in self.benchmark_overrides:
        self.benchmark_config['config'][override]= self.benchmark_overrides[override]
      self.benchmark_config['config']['metrics']=self.metrics
      self.loader = self.step['loader']

    def execute(self):
      loader = Loader(self.db_config, self.dataset_config, self.benchmark_config)
      if "MPLoader" in self.loader:
          logging.info(f"Loading in MPLoader.")
          loader.load_in_mploader()
      elif "RAYLoader" in self.loader:
          logging.info(f"Loading in RAYLoader.")
          loader.load_in_rayloader()

      vecbench_ray = os.getenv("VECBENCH_RAY", "False")
      if "MPLoader" in self.loader or "True" in vecbench_ray:
          # Generate the report for this run.
          generate_report(self.db_config, self.dataset_config, self.benchmark_config, self.report_file, self.report_folder)
