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
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from metrics.metrics import Metrics

logging.getLogger().setLevel(logging.INFO)

class GCPMetrics(Metrics):
    def __init__(self, run_id):
      console_enabled = False
      resource = Resource.create(
          {
              "service.name": "dbloader",
              "service.namespace": "vecbench",
              "service.instance.id": "instance007",
          }
      )

      readers = []

      if console_enabled:
        # Console Exporter
        exporter = ConsoleMetricExporter()
        reader1 = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        readers.append(reader1)

      cloud_exporter = CloudMonitoringMetricsExporter()
      reader2 = PeriodicExportingMetricReader(cloud_exporter, export_interval_millis=5000)
      readers.append(reader2)

      self.provider = MeterProvider(metric_readers=readers, resource=resource)
      metrics.set_meter_provider(self.provider)
      self.provider = metrics.get_meter_provider()
      self.meter = self.provider.get_meter("vecbench")
      self.gcpmetrics = {}

    def getMetric(self, name):
      if name not in self.gcpmetrics:
        self.gcpmetrics[name] = self.meter.create_histogram(
            name=name,
            description=f"Vecbench metric:{name}",
            unit="seconds",
        )
      return self.gcpmetrics[name]


    def collect(self, name, tags, field, val):
      requests_histogram = self.getMetric(field)
      requests_histogram.record(val, attributes=tags)

