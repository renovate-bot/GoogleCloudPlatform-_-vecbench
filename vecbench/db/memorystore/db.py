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

import dataclasses
import random
import redis
import logging
import numpy as np

class Memorystore:
    def __init__(self, config):
        self.type = "Memorystore"
        self.ip = config["ip"]
        self.port = config["port"]
        self.read_ip = ""
        if "read_ip" in config:
            logging.info(f"Setting up Redis read endpoint")
            self.read_ip = config["read_ip"]
            self.read_endpoint = redis.Redis(host=self.read_ip, port=self.port)
        self.read_replicas = 0
        if "read_replicas" in config:
            self.read_replicas = config["read_replicas"]
        self.field_name = "vector"
        self.redis = redis.Redis(host=self.ip, port=self.port)
        self.index_name = "vecbench"

    def load_table(self, table_name):
        pass

    def create_index(self, table_name, db_recreate, vector_dimensions, benchmark_config):
        index_recreate=benchmark_config["config"]["index_recreate"]
        index_type=benchmark_config["config"]["index_type"]
        index_config=benchmark_config["config"]["index_config"]
        algo=benchmark_config["config"]["algo"]

        if index_recreate == False:
            logging.info(f"Skip indexing as index_recreate is set to False")
            return

        if index_type not in ["hnsw", "flat"]:
            logging.info(f"Index type not supported. Only hnsw and flat index types are supported.")
            return

        if index_type == "hnsw":
            ef_construction=index_config["ef_construction"]
            ef_runtime=benchmark_config["config"]["probes"]
            m=index_config["m"]

        initial_cap = 10240
        if "initial_cap" in index_config:
            initial_cap = index_config["initial_cap"]

        distance = ""
        if "vector_cosine_ops" in algo:
            distance = "cosine"
        if "vector_l2_ops" in algo:
            distance = "l2"
        if "vector_ip_ops" in algo:
            distance = "ip"

        # Drop existing index
        args = [
                "FT.DROPINDEX",
                self.index_name,
        ]
        print("Running Redis command:", args)
        try:
            self.redis.execute_command(*args)
        except redis.exceptions.ResponseError as e:
            print("Dropping index failed due to", e)

        # Delete existing data
        args = [
               "FLUSHDB",
               "SYNC",
        ]
        print("Running Redis command:", args)
        self.redis.execute_command(*args)


        # Create index
        if index_type == "hnsw":
            args = [
                "FT.CREATE",
                self.index_name,
                "SCHEMA",
                self.field_name,
                "VECTOR",
                "HNSW",
                "14",  # number of remaining arguments
                "TYPE",
                "FLOAT32",
                "DIM",
                vector_dimensions,
                "DISTANCE_METRIC",
                distance,
                "M",
                m,
                "EF_CONSTRUCTION",
                ef_construction,
                "EF_RUNTIME",
                ef_runtime,
                "INITIAL_CAP",
                initial_cap,
                ]
        else:
            args = [
                "FT.CREATE",
                self.index_name,
                "SCHEMA",
                self.field_name,
                "VECTOR",
                "FLAT",
                "8",  # number of remaining arguments
                "TYPE",
                "FLOAT32",
                "DIM",
                vector_dimensions,
                "DISTANCE_METRIC",
                distance,
                "INITIAL_CAP",
                initial_cap,
            ]

        print("Running Redis command:", args)
        self.redis.execute_command(*args)


    def configure_search_session(self, benchmark_config):
        self.ef_runtime = benchmark_config['probes']
        if benchmark_config['index_type'] not in ["hnsw", "flat"]:
            logging.info(f"Index type not supported. Only hnsw and flat index types are supported.")
        self.index_type = benchmark_config['index_type']

    def set_value(self, table_name):
        pass

    def populate(self, table_name, data, start):
        if len(data[0]) == 2:
            self.populate_with_id(table_name, data)
        else:
            self.populate_without_id(table_name, data, start)

    def populate_with_id(self, table_name, data):
        p = self.redis.pipeline(transaction=False)
        for i, (id, embedding) in enumerate(data):
            p.execute_command("HSET", id, self.field_name, embedding.astype(np.float32).tobytes())
            if i % 1000 == 999:
                p.execute()
                p.reset()
        p.execute()

    def populate_without_id(self, table_name, data, start_id):
        p = self.redis.pipeline(transaction=False)
        for i, embedding in enumerate(data):
            p.execute_command("HSET", start_id + i, self.field_name, embedding.astype(np.float32).tobytes())
            if i % 1000 == 999:
                p.execute()
                p.reset()
        p.execute()

    def index_embeddings(self, benchmark_config, vector_table):
        pass

    def returned_rows(self, response):
        return response

    def get_by_id(self, id):
        q = [
            "HGET",
            id,
            self.field_name
        ]
        return ((id, np.frombuffer(self.redis.execute_command(*q), dtype=np.float32)),)


    def annsearch(self, embedding, limit, algo):
        if self.index_type == "hnsw":
            q = [
                "FT.SEARCH",
                self.index_name,
                f"*=>[KNN {limit} @{self.field_name} $BLOB EF_RUNTIME {self.ef_runtime}]",
                "NOCONTENT",
                "LIMIT",
                "0",
                str(limit),
                "PARAMS",
                "2",
                "BLOB",
                embedding.astype(np.float32).tobytes(),
                "DIALECT",
                "2",
            ]
        else:
            q = [
                "FT.SEARCH",
                self.index_name,
                f"*=>[KNN {limit} @{self.field_name} $BLOB]",
                "NOCONTENT",
                "LIMIT",
                "0",
                str(limit),
                "PARAMS",
                "2",
                "BLOB",
                embedding.astype(np.float32).tobytes(),
                "DIALECT",
                "2",
            ]

        # Send the search query to the primary if no read replicas are available. If read replicas
        # are provisioned, distribute traffic between the primary and read endpoints.
        if self.read_ip == "" or random.randint(0, self.read_replicas) == 0:
            redis_endpoint = self.redis
        else:
            redis_endpoint = self.read_endpoint

        try:
            return [(int(doc),) for doc in redis_endpoint.execute_command(*q)[1:]]
        except redis.exceptions.ResponseError as e:
            print("FT.SEARCH failed for vector", embedding, "query ", q, " with error ", e)
            return []

    def anninsert(self, embedding, table_name, insert_id=None):
        if insert_id != None:
            id = insert_id
        else:
            logging.info(f"Insert id not provided, skipping insert operation")
            return
        return self.redis.execute_command("HSET", id, self.field_name, np.array(embedding).astype(np.float32).tobytes())

    def annupdate(self, id, embedding, table_name):
        return self.redis.execute_command("HSET", id, self.field_name, np.array(embedding).astype(np.float32).tobytes())

    def anndelete(self, id, table_name):
        p = self.redis.pipeline(transaction=True)
        p.execute_command("EXISTS", id)
        p.execute_command("DEL", id)
        res = p.execute()
        return DeleteResponse(rowcount=res[0])

    def anndatasetsize(self, table_name):
        return self.redis.execute_command("DBSIZE")

    def get_by_id_batch(self, ids):
        vectors = []
        if (len(ids) > 100):
            for i in range(0, len(ids), 100):
                vectors.append(get_by_id_batch(ids[i, i+100]))
            return vectors

        p = self.redis.pipeline(transaction=False)
        for id in ids:
            p.execute_command("HGET", id, self.field_name)
        results = p.execute()
        for i in range(0, len(results)):
            vectors.append(((ids[i], np.frombuffer(results[i], dtype=np.float32)),))
        return vectors

@dataclasses.dataclass
class DeleteResponse:
    rowcount: int
