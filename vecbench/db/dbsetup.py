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

from db.alloydb.db import AlloyDB
from db.csqlpg.db import CsqlPG
# Firestore currently requires a custom SDK to work
# from db.firestore.db import Firestore
from db.pinecone.db import Pinecone
from db.vectorsearch.db import VertexVectorSearch
from db.memorystore.db import Memorystore
from db.spanner.db import Spanner
from db.mysql.db import CsqlMySQL
from db.milvus.db import Milvus
import time

class DBSetup:
    def __init__(self, db_config):
        self.type = db_config["type"]
        config = db_config["config"]
        if self.type == "AlloyDB":
            self.db = AlloyDB(config)
        if self.type == "CsqlPG":
            self.db = CsqlPG(config)
        if self.type == "AlloyDBOmni":
            self.db = AlloyDB(config)
        # if self.type == "Firestore":
            # self.db = Firestore(config)
        if self.type == "Pinecone":
            self.db = Pinecone(config)
        if self.type == "VertexVectorSearch":
            self.db = VertexVectorSearch(config)
        if self.type == "Memorystore":
              self.db = Memorystore(config)
        if self.type == "Spanner":
            self.db = Spanner(config)
        if self.type == "CsqlMySQL":
            self.db = CsqlMySQL(config)
        if self.type == "Milvus":
            self.db = Milvus(config)

    def load_dataset(self, table_name, db_dataset, start, end, algo):
        assert len(db_dataset) == (end - start)
        if self.type in ["Pinecone", "VertexVectorSearch"]:
            self.db.populate(table_name, db_dataset, start, algo)
        else:
            self.db.populate(table_name, db_dataset, start)

    def load_table(self, table_name, algo = None):
        if self.type in ["Pinecone", "VertexVectorSearch"]:
            self.vector_table = self.db.load_index(table_name, algo)
        else:
            self.vector_table = self.db.load_table(table_name)

    def create_table(self, table_name, db_recreate, vector_dimension, benchmark_config):
        if self.type in ["Pinecone", "Memorystore", "VertexVectorSearch"]:
            self.vector_table = self.db.create_index(table_name, db_recreate, vector_dimension, benchmark_config)
        else:
            self.vector_table = self.db.CreateTable(table_name, db_recreate, vector_dimension)

    def configure_search_session(self, benchmark_config):
        self.db.configure_search_session(benchmark_config)

    def index_dataset(self, benchmark_config):
        start = time.time()
        self.db.index_embeddings(
            benchmark_config,
            self.vector_table,
        )
        end = time.time()
        print(f"Indexing took {end-start} seconds.")

    def set_value(self, table_name):
        return self.db.set_value(table_name)

    def annsearch(self, embedding, limit, algo):
        return self.db.annsearch(embedding=embedding, limit=limit, algo=algo)

    def annbatchsearch(self, embeddings, limit, algo):
        return self.db.annbatchsearch(embeddings=embeddings, limit=limit, algo=algo)

    def anninsert(self, embedding, table_name, insert_id=None):
        return self.db.anninsert(embedding=embedding, table_name = table_name, insert_id=insert_id)

    def annupdate(self, id, embedding, table_name):
        return self.db.annupdate(id=id, embedding=embedding, table_name = table_name)

    def anndelete(self, id, table_name):
        return self.db.anndelete(id=id, table_name = table_name)

    def annfilteredsearch(self, id, embedding, limit, algo):
        return self.db.annfilteredsearch(id=id, embedding=embedding, limit=limit, algo=algo)

    def anndatasetsize(self, table_name):
        return self.db.anndatasetsize(table_name=table_name)

    def returned_rows(self, response):
        return self.db.returned_rows(response)

    def returned_rows_batch(self, response):
        return self.db.returned_rows_batch(response)

    def get_by_id(self, id):
        return self.db.get_by_id(id)

    def get_by_id_batch(self, ids):
        return self.db.get_by_id_batch(ids)
