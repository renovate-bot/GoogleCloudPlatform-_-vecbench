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

from jinja2 import Environment, FileSystemLoader, Template
from report.template_functions import config_template

def render(df, benchmark_config, report_file):
    template_file= benchmark_config['config']['report_template']
    if template_file is None:
        print("No template for benchmar.")
        return

    environment = Environment(loader=FileSystemLoader("config/benchmark/templates/"))
    template = environment.get_template(template_file)
    config_template(template)
    report = template.render(df=df, benchmark_config=benchmark_config)
    print(report)
    if report_file is not None:
        with open(report_file, "w") as rfile:
            rfile.write(report)