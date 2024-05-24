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
import asyncio
from datetime import datetime

from tinyflux import TinyFlux
from tinyflux import Point as TPoint
import os, errno
from metrics.metrics import Metrics

class PandasMetrics(Metrics):
    def __init__(self, run_id):
        download_directory = "downloads"
        try:
            os.makedirs(download_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.db = TinyFlux(
            f"{download_directory}/db_{run_id}_{os.getpid()}.csv", flush_on_insert=False
        )
        asyncio.run(self.setup())

    async def setup(self):
        self.queue = asyncio.Queue()

    async def process(self):
        self.task = asyncio.create_task(self.worker())


    def collect(self, name, tags, field, val):
        point = TPoint(time=datetime.now(), measurement=name, tags=tags, fields={field: val})
        self.queue.put_nowait(point)

    async def worker(self):
        while True:
            # Get a "Point" out of the queue.
            point = await self.queue.get()
            self.db.insert(point)
            self.queue.task_done()

    def close(self):
        asyncio.run(self.process())
        self.db.close()
