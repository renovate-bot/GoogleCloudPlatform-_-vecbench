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

from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Integer,
    Index,
    MetaData,
    inspect,
    text,
    select,
    ARRAY,
    FLOAT
)
from sqlalchemy.orm import sessionmaker
import metrics
import time
import logging
import os
from db.dbglobal import DBGlobal

# 'sqlalchemy.engine' to see sql log
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()

class Spanner:
    def __init__(self, config):
        self.dataset_name = None
        self.vector_table = None
        db_config = f"spanner+spanner:///projects/{config['project_id']}/instances/{config['instance-id']}/databases/{config['database_id']}"
        self.type = "Spanner"
        self.engine = create_engine(db_config, pool_pre_ping=True)
        autocommit_read_engine = self.engine.execution_options(isolation_level="AUTOCOMMIT", read_only=True)
        self._sessionclass = sessionmaker(bind=autocommit_read_engine)
        self.search_session = self._sessionclass()
        self.table_exists = False
        metrics_type = metrics.NOOP_METRICS
        run_id = config['run_id']
        self.metrics = metrics.get_metrics(metrics_type, run_id)


    def load_table(self, table_name): 
        metadata = MetaData()
        vector_table = Table(table_name, metadata, autoload_with=self.engine)
        self.vector_table = vector_table
        print(vector_table.c)
        self.table_exists = True
        return vector_table

    def CreateTable(self, table_name, db_recreate, vector_dimensions):
        """Creates a database and tables for sample data."""
        metadata = MetaData()
        vector_table = Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("embeddings", ARRAY(FLOAT)),
        )

        metadata.create_all(self.engine)

        if db_recreate:
            with self.engine.begin() as conn:
                conn.execute(vector_table.delete())
            self.table_exists = False

        self.vector_table = vector_table
        if inspect(self.engine).has_table(table_name):
            self.table_exists = True
            logging.info("Table {} successfully created.".format(vector_table.name))
        else:
            logging.warning("Table {} did not created.".format(vector_table.name))
        return vector_table
    
    def populate(self, table_name, data, start):
        pid = os.getpid()
        self.dataset_name = table_name
        logging.info(f"Populating table:{table_name} with size {len(data)} with id status: {len(data[0]) == 2}")
        tags = {
            "tool": "db.populate",
            "type": self.type,
            "pid": pid,
            "dataset_name": table_name,
        }
        ops_limit = 10000
        num_transcation = int(len(data) / ops_limit) + 1
        populate_with_id = len(data[0]) == 2
        for i in range(0, num_transcation):
            ll = i * ops_limit
            rl = (i+1) * ops_limit
            if i == num_transcation - 1:
                rl = len(data)
                if rl == ll: 
                    break
            with self.engine.begin() as conn:
            # Few datasets have ID fields explicitly and its important to
            # retain them as recall calculation use neighbour ID.
                if populate_with_id:
                    self.populate_with_id(conn, table_name, data[ll: rl], tags)
                else:
                    self.populate_without_id(conn, table_name, data[ll: rl], start+ll, tags)
            logging.info(f"PID:{pid}, transcation number:{i} committed with size {rl-ll}")
        logging.info(f"Populating table PID:{pid} of table {table_name} committed with size {len(data)}")

    def populate_with_id(self, conn, table_name, data, tags):
        for i, (id, embedding) in enumerate(data):
            embedding = [float(x) for x in embedding]
            start = time.time()
            conn.execute(
                text(f"INSERT INTO {table_name} (id, embeddings) VALUES (:id, :embedding)"), 
                {"id":id, "embedding":embedding}
            )
            end = time.time()
            self.metrics.collect("insert", tags, "elapsed", (end - start))
            self.metrics.collect("insert", tags, "inserted", i)

    def populate_without_id(self, conn, table_name, data, start_id, tags):
        for i, embedding in enumerate(data):
            embedding = [float(x) for x in embedding]
            start = time.time()
            conn.execute(
                text(f"INSERT INTO {table_name} (id, embeddings) VALUES (:id, :embedding)"), 
                {"id":start_id + i, "embedding":embedding}
            )
            end = time.time()
            self.metrics.collect("insert", tags, "elapsed", (end - start))
            self.metrics.collect("insert", tags, "inserted", i)

    def index_embeddings(self, benchmark_config, vector_table):
        pass

    def configure_search_session(self, benchmark_config):
        pass

    def set_value(self, table_name):
        pass

    def get_by_id(self, id):
        rows = self.search_session.execute(
            select(self.vector_table)
            .where(self.vector_table.columns.id == id)
        ).fetchall()
        return rows
    
    def annsearch(self, embedding, limit, algo):
        embedding = self.embedding_to_float(embedding)
        if algo == DBGlobal.COSINE_SIMILARITY:
            method = "COSINE_DISTANCE"
        elif algo == DBGlobal.L2_DISTANCE:
            method = "EUCLIDEAN_DISTANCE"
        else:
            return None
        statement = f"SELECT id FROM {self.vector_table.name} ORDER BY {method}(embeddings, :embedding) LIMIT :limit"
        return self.search_session.execute(
            text(statement), {"embedding":embedding, "limit":limit}
        )

    def anninsert(self, embedding, table_name, insert_id=None):
        embedding = self.embedding_to_float(embedding)
        end_cur = self.max_id(table_name)
        rows = end_cur.fetchall()
        end = rows[0][0]
        id = 1
        if end is not None:
            id = end + 1
        with self.engine.begin() as conn:
            return conn.execute(
                text(f"INSERT INTO {table_name} (id, embeddings) VALUES (:id, :embedding)"),
                {"id":id, "embedding":embedding}
            )

    def annupdate(self, id, embedding, table_name):
        embedding = self.embedding_to_float(embedding)
        with self.engine.begin() as conn:
            return conn.execute(
                text(f"UPDATE {table_name} SET embeddings=:embedding WHERE id=:id"),
                {"id":id, "embedding":embedding}
            )

    def anndelete(self, id, table_name):
        with self.engine.begin() as conn:
            return conn.execute(
                text(f"DELETE FROM {table_name} WHERE id=:id"),
                {"id":id}
            )
    
    def annfilteredsearch(self, id, embedding, limit, algo):
        embedding = self.embedding_to_float(embedding)
        if algo == DBGlobal.COSINE_SIMILARITY:
            method = "COSINE_DISTANCE"
        elif algo == DBGlobal.L2_DISTANCE:
            method = "EUCLIDEAN_DISTANCE"
        else:
            return None
        statement = f"SELECT id FROM {self.vector_table.name} WHERE id < :id ORDER BY {method}(embeddings, :embedding) LIMIT :limit"
        return self.search_session.execute(
            text(statement), {"id":id, "embedding":embedding, "limit":limit}
        )

    def annfilteredrangesearch(self, id_start, id_end, embedding, limit, algo):
        embedding = self.embedding_to_float(embedding)
        if algo == DBGlobal.COSINE_SIMILARITY:
            method = "COSINE_DISTANCE"
        elif algo == DBGlobal.L2_DISTANCE:
            method = "EUCLIDEAN_DISTANCE"
        else:
            return None
        statement = f"SELECT id FROM {self.vector_table.name} WHERE id >= :id_start and " \
                    f"id < :id_end ORDER BY {method}(embeddings, :embedding) LIMIT :limit"        
        return self.search_session.execute(
            text(statement), {"id_start":id_start, "id_end":id_end, "embedding":embedding, "limit":limit}
        )
    
    def annfilteredmodsearch(self, mod, remainder, embedding, limit, algo):
        embedding = self.embedding_to_float(embedding)
        if algo == DBGlobal.COSINE_SIMILARITY:
            method = "COSINE_DISTANCE"
        elif algo == DBGlobal.L2_DISTANCE:
            method = "EUCLIDEAN_DISTANCE"
        else:
            return None
        statement = f"SELECT id FROM {self.vector_table.name} WHERE MOD(id, :mod)=:remainder " \
                    f"ORDER BY {method}(embeddings, :embedding) LIMIT :limit"        
        return self.search_session.execute(
            text(statement), {"mod":mod, "remainder":remainder, "embedding":embedding, "limit":limit}
        )


    def anndatasetsize(self, table_name):
        return self.search_session.execute(text(f"SELECT COUNT(id) FROM {table_name}")).fetchall()[0][0]

    def returned_rows(self, response):
        return response.fetchall()
    
    def embedding_to_float(self, embedding):
        if isinstance(embedding[0], int):
            return [float(x) for x in embedding]
        return embedding
    
    def max_id(self, table_name):
        return self.search_session.execute(text(f"SELECT MAX(id) FROM {table_name}"))

    def get_by_id_batch(self, ids):
        vectors = []
        for id in ids:
            vectors.append(self.get_by_id(id))
        return vectors
