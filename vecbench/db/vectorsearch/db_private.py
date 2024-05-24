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

import grpc
import numpy as np
from db.vectorsearch.common import api_backoff
from google.cloud.aiplatform.matching_engine._protos import match_service_pb2
from google.cloud.aiplatform.matching_engine._protos import match_service_pb2_grpc

class VectorSearchPrivate:
    def __init__(self, config):
        self.deployed_index = config["deployed_index_id"]
        if "frac_leaf_nodes_to_search" in config:
            self.frac_leaf_nodes_to_search = float(config["frac_leaf_nodes_to_search"])
        else:
            self.frac_leaf_nodes_to_search = 0.05
        host = config["private_endpoint_config"]["grpc_address"]
        channel = grpc.insecure_channel(host)
        self.stub = match_service_pb2_grpc.MatchServiceStub(channel)

    def returned_rows(self, response):
        res = []
        for neighbor in response.neighbor:
            res.append((int(neighbor.id), ))
        return res

    def returned_rows_batch(self, response):
        res = []
        batch_match_response = response.responses[0]
        for match_resp in batch_match_response.responses:
            neighbors = []
            for neighbor in match_resp.neighbor:
                neighbors.append((int(neighbor.id), ))
            res.append(neighbors)
        return res

    @api_backoff
    def get_by_id(self, id):
        req = match_service_pb2.BatchGetEmbeddingsRequest(
            deployed_index_id=self.deployed_index,
            id=[str(id)]
        )
        resp = self.stub.BatchGetEmbeddings(req)
        return [(id, np.array(resp.embeddings[0].float_val, dtype=np.float32))]

    @api_backoff
    def get_by_id_batch(self, ids):
        req = match_service_pb2.BatchGetEmbeddingsRequest(
            deployed_index_id=self.deployed_index,
            id=[str(id) for id in ids]
        )
        resp = self.stub.BatchGetEmbeddings(req)
        res = []
        for i in range(len(ids)):
            res.append([(ids[i], np.array(resp.embeddings[i].float_val, dtype=np.float32))])
        return res

    @api_backoff
    def annsearch(self, embedding, limit, algo):
        req = match_service_pb2.MatchRequest(
            num_neighbors=limit,
            deployed_index_id=self.deployed_index,
            float_val=embedding,
            fraction_leaf_nodes_to_search_override=self.frac_leaf_nodes_to_search
        )
        return self.stub.Match(req)

    @api_backoff
    def annbatchsearch(self, embeddings, limit, algo):
        req = match_service_pb2.BatchMatchRequest()
        batch_match_request_per_index = req.requests.add()
        batch_match_request_per_index.deployed_index_id = self.deployed_index
        for embedding in embeddings:
            match_req = match_service_pb2.MatchRequest(
                num_neighbors=limit,
                deployed_index_id=self.deployed_index,
                float_val=embedding,
                fraction_leaf_nodes_to_search_override=self.frac_leaf_nodes_to_search
            )
            batch_match_request_per_index.requests.append(match_req)

        return self.stub.BatchMatch(req)