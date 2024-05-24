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

import google.auth
import grpc
import numpy as np
from db.vectorsearch.common import api_backoff
from google.cloud.aiplatform_v1 import IndexDatapoint
from google.cloud.aiplatform_v1 import FindNeighborsRequest
from google.cloud.aiplatform_v1 import MatchServiceClient
from google.cloud.aiplatform_v1 import ReadIndexDatapointsRequest
from google.cloud.aiplatform_v1.services.match_service.transports import grpc as match_transports_grpc

class VectorSearchPublic:
    def __init__(self, config):
        self.deployed_index = config["deployed_index_id"]
        if "frac_leaf_nodes_to_search" in config:
            self.frac_leaf_nodes_to_search = float(config["frac_leaf_nodes_to_search"])
        else:
            self.frac_leaf_nodes_to_search = 0.05
        public_endpoint_config = config["public_endpoint_config"]
        self.public_endpoint = public_endpoint_config["public_endpoint_url"]
        credentials, _ = google.auth.default()
        request = google.auth.transport.requests.Request()
        channel = google.auth.transport.grpc.secure_authorized_channel(
            credentials,
            request,
            self.public_endpoint,
            ssl_credentials=grpc.ssl_channel_credentials(),
        )
        self.client = MatchServiceClient(
            transport=match_transports_grpc.MatchServiceGrpcTransport(
                channel=channel,
            ),
        )
        self.index_endpoint = public_endpoint_config["index_endpoint"]

    def returned_rows(self, response):
        res = []
        for neighbor in response.nearest_neighbors[0].neighbors:
            res.append((int(neighbor.datapoint.datapoint_id), ))
        return res

    def returned_rows_batch(self, response):
        res = []
        for query_response in response.nearest_neighbors:
            neighbors = []
            for neighbor in query_response.neighbors:
                neighbors.append((int(neighbor.datapoint.datapoint_id), ))
            res.append(neighbors)
        return res

    @api_backoff
    def get_by_id(self, id):
        req = ReadIndexDatapointsRequest(
            index_endpoint=self.index_endpoint,
            deployed_index_id=self.deployed_index,
            ids=[str(id)]
        )
        resp = self.client.read_index_datapoints(req)
        return [(id, np.array(resp.datapoints[0].feature_vector, dtype=np.float32))]

    @api_backoff
    def get_by_id_batch(self, ids):
        req = ReadIndexDatapointsRequest(
            index_endpoint=self.index_endpoint,
            deployed_index_id=self.deployed_index,
            ids=[str(id) for id in ids]
        )
        resp = self.client.read_index_datapoints(req)
        res = []
        for i in range(len(ids)):
            res.append([(ids[i], np.array(resp.datapoints[i].feature_vector, dtype=np.float32))])
        return res

    @api_backoff
    def annsearch(self, embedding, limit, algo):
        request = FindNeighborsRequest(
            index_endpoint=self.index_endpoint,
            deployed_index_id=self.deployed_index,
        )
        dp1 = IndexDatapoint(
            datapoint_id="0",
            feature_vector=embedding,
        )
        query = FindNeighborsRequest.Query(
            datapoint=dp1,
            neighbor_count=limit,
            fraction_leaf_nodes_to_search_override=self.frac_leaf_nodes_to_search,
        )
        request.queries.append(query)
        return self.client.find_neighbors(request)

    @api_backoff
    def annbatchsearch(self, embeddings, limit, algo):
        request = FindNeighborsRequest(
            index_endpoint=self.index_endpoint,
            deployed_index_id=self.deployed_index,
        )
        for embedding in embeddings:
            dp = IndexDatapoint(
                datapoint_id="0",
                feature_vector=embedding,
            )
            query = FindNeighborsRequest.Query(
                datapoint=dp,
                neighbor_count=limit,
                fraction_leaf_nodes_to_search_override=self.frac_leaf_nodes_to_search,
            )
            request.queries.append(query)
        return self.client.find_neighbors(request)