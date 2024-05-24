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
import uuid
from db.vectorsearch.db_public import VectorSearchPublic
from db.vectorsearch.db_private import VectorSearchPrivate
from google.cloud.aiplatform_v1 import GetIndexRequest
from google.cloud.aiplatform_v1 import IndexDatapoint
from google.cloud.aiplatform_v1 import IndexServiceClient
from google.cloud.aiplatform_v1 import RemoveDatapointsRequest
from google.cloud.aiplatform_v1 import UpsertDatapointsRequest

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()

class VertexVectorSearch:
    def __init__(self, config):
        self.type = "VertexVectorSearch"
        self.index = config["index"]
        self.index_client = IndexServiceClient()
        self.networking_type = config["networking_type"]
        if self.networking_type == "public":
            self.read_client = VectorSearchPublic(config)
        elif self.networking_type == "private":
            self.read_client = VectorSearchPrivate(config)

    def returned_rows(self, response):
        return self.read_client.returned_rows(response)

    def returned_rows_batch(self, response):
        return self.read_client.returned_rows_batch(response)

    def get_by_id(self, id):
        return self.read_client.get_by_id(id)

    def get_by_id_batch(self, ids):
        return self.read_client.get_by_id_batch(ids)
    def annsearch(self, embedding, limit, algo):
        return self.read_client.annsearch(embedding=embedding, limit=limit, algo=algo)

    def annbatchsearch(self, embeddings, limit, algo):
        return self.read_client.annbatchsearch(embeddings=embeddings, limit=limit, algo=algo)

    def anninsert(self, embedding, table_name, insert_id=None):
        req = UpsertDatapointsRequest(
            index=self.index,
        )
        dp1 = IndexDatapoint(
            datapoint_id= str(insert_id) if insert_id is not None else uuid.uuid4(),
            feature_vector=embedding,
        )
        req.datapoints.append(dp1)
        return self.index_client.upsert_datapoints(req)

    def annupdate(self, id, embedding, table_name):
        req = UpsertDatapointsRequest(
            index=self.index,
        )
        dp1 = IndexDatapoint(
            datapoint_id=str(id),
            feature_vector=embedding,
        )
        req.datapoints.append(dp1)
        return self.index_client.upsert_datapoints(req)

    def anndelete(self, id, table_name):
        req = RemoveDatapointsRequest(
            index=self.index,
        )
        req.datapoints.append(str(id))
        return self.index_client.remove_datapoints(req)

    def anndatasetsize(self, table_name):
        req = GetIndexRequest(
            name=self.index
        )
        resp = self.index_client.get_index(req)
        return int(resp.index_stats.vectors_count)

    def load_index(self, table_name, algo):
        pass

    def configure_search_session(self, benchmark_config):
        pass

    def set_value(self, table_name):
        pass

    def index_embeddings(self, benchmark_config, vector_table):
        pass

    def create_index(self, table_name, db_recreate, vector_dimension, benchmark_config):
        pass

    def populate(self, table_name, data, start):
        pass
