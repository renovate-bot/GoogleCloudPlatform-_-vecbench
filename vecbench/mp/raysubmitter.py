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

from ray.scripts.scripts import cli
from ray.job_submission import JobSubmissionClient, JobStatus
import time
import asyncio
import os
from google.cloud import compute_v1
from pyaml_env import parse_config

RAY_GCP_CONFIG = "config/ray/vecbenchray_gcp.yaml"


async def tail_logs(client, job_id):
    async for lines in client.tail_job_logs(job_id):
        print(lines, end="")


class RayLoader:
    def get_ray_head_ip(self, project_id, availability_zone):
        client = compute_v1.InstancesClient()
        request = compute_v1.ListInstancesRequest(
            project=project_id,
            zone=availability_zone,
        )
        page_result = client.list(request=request)

        for response in page_result:
            if "head" in response.name:
                head_ip = response.network_interfaces[0].network_i_p
                return head_ip
        return None

    def wait_until_status(
        self, client, job_id, status_to_wait_for, timeout_seconds=600
    ):
        start = time.time()
        while time.time() - start <= timeout_seconds:
            status = client.get_job_status(job_id)
            print(f"status: {status}")
            if status in status_to_wait_for:
                break
            time.sleep(1)

    def ray_submit(self, argv):
        env_vars = {}
        for name, value in os.environ.items():
            if "VECBENCH" in name:
                env_vars[name] = value
        env_vars["VECBENCH_RAY"] = "True"

        rayconfig = parse_config(RAY_GCP_CONFIG)
        project_id = rayconfig["provider"]["project_id"]
        availability_zone = rayconfig["provider"]["availability_zone"]
        head_ip = self.get_ray_head_ip(project_id, availability_zone)
        client = JobSubmissionClient(f"http://{head_ip}:8265")
        job_id = client.submit_job(
            # Entrypoint shell command to execute
            entrypoint=f"python vecbench.py {' '.join(argv[1:])}",
            # Path to the local directory that contains the script.py file
            runtime_env={
                "working_dir": ".",
                "excludes": ["venv", "downloads"],
                "env_vars": env_vars,
                "pip": "requirements.txt",
            },
        )

        asyncio.run(tail_logs(client, job_id))

        self.wait_until_status(
            client, job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}
        )

    def ray_down(self):
        cli(["down", "--yes", RAY_GCP_CONFIG], standalone_mode=False)

    def ray_launch(self):
        try:
            cli(["up", "--yes", RAY_GCP_CONFIG])
        except SystemExit as e:
            if e.code != 0:
                raise
