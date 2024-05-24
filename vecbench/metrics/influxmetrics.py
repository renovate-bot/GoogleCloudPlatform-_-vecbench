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
from influxdb_client import InfluxDBClient, WriteOptions, Point
import os
import logging
import os
from metrics.metrics import Metrics

logging.getLogger().setLevel(logging.INFO)


class InfluxMetrics(Metrics):
    def __init__(self, run_id):
        self.bucket = os.getenv("VECBENCH_BUCKET")
        self.org = os.getenv("VECBENCH_ORG")
        self.token = os.getenv("VECBENCH_TOKEN")
        self.url = os.getenv("VECBENCH_INFLUX_URL")
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(
            write_options=WriteOptions(
                batch_size=1000,
                flush_interval=1_000,
                jitter_interval=2_000,
                retry_interval=5_000,
                max_retries=5,
                max_retry_delay=30_000,
                max_close_wait=300_000,
            )
        )

    def collect(self, name, tag, tagval, field, val):
        p = Point(name).tag(tag, tagval).field(field, val)
        self.write_api.write(bucket=self.bucket, org=self.org, record=p)

    def collect(self, name, tags, field, val):
        p = Point(name).field(field, val)
        for tag, tagval in tags.items():
            p.tag(tag, tagval)
        self.write_api.write(bucket=self.bucket, org=self.org, record=p)

    def close(self):
        self.write_api.close()
