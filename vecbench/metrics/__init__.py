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

from metrics.metrics import Metrics
from metrics.pandasmetrics import PandasMetrics
from metrics.influxmetrics import InfluxMetrics
from metrics.gcpmetrics import GCPMetrics

NOOP_METRICS = "NOOP_METRICS"
PANDAS_METRICS = "PANDAS_METRICS"
INFLUX_METRICS = "INFLUX_METRICS"
GCP_METRICS = "GCP_METRICS"

def get_metrics(Type=None, run_id=None):
    if Type == NOOP_METRICS:
        return Metrics(run_id)
    if Type == PANDAS_METRICS:
        return PandasMetrics(run_id)
    if Type == INFLUX_METRICS:
        return InfluxMetrics(run_id)
    if Type == GCP_METRICS:
        return GCPMetrics(run_id)