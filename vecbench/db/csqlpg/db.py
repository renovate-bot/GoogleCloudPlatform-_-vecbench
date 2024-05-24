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
)
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
from pgvector.psycopg2 import register_vector
import metrics
import time
import logging
import os
from db.dbglobal import DBGlobal

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()

# Enable below to verify SQLAlchemy's commands.
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

class CsqlPG:
    def __init__(self, config):
        db_config = f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['ip']}:{config['port']}/{config['database']}"
        self.type = "CsqlPG"
        self.engine = create_engine(db_config, pool_pre_ping=True, isolation_level="AUTOCOMMIT")
        self._sessionclass = sessionmaker(bind=self.engine)
        self.search_session = self._sessionclass()
        self.table_exists = False
        metrics_type = metrics.NOOP_METRICS
        run_id = config['run_id']
        self.metrics = metrics.get_metrics(metrics_type, run_id)


    def load_table(self, table_name):
        metadata = MetaData()
        vector_table = Table(table_name, metadata, autoload_with=self.engine)
        self.vector_table = vector_table
        print (vector_table.c)
        return vector_table

    def CreateTable(self, table_name, db_recreate, vector_dimensions):
        if inspect(self.engine).has_table(table_name):
            self.table_exists = True

        metadata = MetaData()
        vector_table = Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("embeddings", Vector(vector_dimensions)),
        )

        if db_recreate == True:
            metadata.drop_all(self.engine)
            self.table_exists = False

        metadata.create_all(self.engine)
        self.vector_table = vector_table
        return vector_table

    def populate(self, table_name, data, start):
        pid = os.getpid()
        self.dataset_name = table_name
        if self.table_exists == False:
            logging.info(f"Populating table:{table_name} with {len(data)}")
            tags = {
                "tool": "db.populate",
                "type": self.type,
                "pid": pid,
                "dataset_name": table_name,
            }
            conn = self.engine.pool._creator()
            register_vector(conn)
            cursor = conn.cursor()
            # Few datasets have ID fields explicitly and its important to
            # retain them as recall calculation use neighbour ID.
            if len(data[0]) == 2:
                self.populate_with_id(cursor, table_name, data, tags)
            else:
                self.populate_without_id(cursor, table_name, data, start, tags)
            conn.commit()

    def populate_with_id(self, cursor, table_name, data, tags):
        for i, (id, embedding) in enumerate(data):
            start = time.time()
            cursor.execute(
                f"INSERT INTO {table_name} (id, embeddings) VALUES (%s, %s)",
                (id, embedding,),
            )
            end = time.time()
            self.metrics.collect("insert", tags, "elapsed", (end - start))
            self.metrics.collect("insert", tags, "inserted", i)

    def populate_without_id(self, cursor, table_name, data, start_id, tags):
        for i, embedding in enumerate(data):
            start = time.time()
            cursor.execute(
                f"INSERT INTO {table_name} (id, embeddings) VALUES (%s, %s)",
                (start_id + i, embedding,),
            )
            end = time.time()
            self.metrics.collect("insert", tags, "elapsed", (end - start))
            self.metrics.collect("insert", tags, "inserted", i)

    def index_embeddings(self, benchmark_config, vector_table):
        index_recreate=benchmark_config["config"]["index_recreate"]
        index_type=benchmark_config["config"]["index_type"]
        index_config=benchmark_config["config"]["index_config"]
        algo=benchmark_config["config"]["algo"]

        if index_recreate == True:
            logging.info(
                f"Indexing table type:{index_type} with algo: {algo} config: {index_config}"
            )
            # Drop existing index if we have one.
            session = self._sessionclass()
            session.execute(text(f"drop index if exists {vector_table}_v_index;"))
            session.commit()

            if "scann" in index_type:
                distance = ""
                if "vector_cosine_ops" in algo:
                    distance = "cosine"
                if "vector_l2_ops" in algo:
                    distance = "l2"
                if "vector_ip_ops" in algo:
                    distance = "ip"
                idx = f"CREATE INDEX {vector_table}_v_index ON {vector_table} USING scann (embeddings {distance}) WITH {index_config};"
                session.execute(text(idx))
                session.commit()
            else:
                index = Index(
                    f"{vector_table}_v_index",
                    vector_table.c.embeddings,
                    postgresql_using=index_type,
                    postgresql_ops={"embeddings": algo},
                    postgresql_with=index_config,
                )
                index.create(self.engine)
            session.execute(text(f"VACUUM (DISABLE_PAGE_SKIPPING) {vector_table};"))

    def configure_search_session(self, benchmark_config):
        index_type = benchmark_config['index_type']
        probes = benchmark_config['probes']
        if index_type == "ivfflat":
            self.search_session.execute(text(f"SET ivfflat.probes = {probes};"))
        elif index_type == "ivf":
            self.search_session.execute(text(f"SET ivf.probes = {probes};"))
        elif index_type == "hnsw":
            self.search_session.execute(text(f"SET  hnsw.ef_search = {probes};"))
        elif index_type == "scann":
            self.search_session.execute(text(f"SET scann.num_leaves_to_search = {benchmark_config['num_leaves_to_search']};"))
        else:
            raise RuntimeError(f"unknown index type {index_type}")
        if index_type in ["ivfflat", "ivf", "hnsw"]:
            self.search_session.execute(
                text(f"SET max_parallel_workers_per_gather = {probes};")
            )

    def set_value(self, table_name):
        end = self.anndatasetmaxid(table_name)
        val = end + 1;
        return self.search_session.execute(text(f"select setval('{table_name}_id_seq',{val})"))

    def get_by_id(self, id):
        return self.search_session.execute(
            select(self.vector_table)
            .where(self.vector_table.columns.id == id)
        ).fetchall()

    def annsearch(self, embedding, limit, algo):
        if algo == DBGlobal.L2_DISTANCE:
            return self.search_session.execute(
                select(self.vector_table.columns.id)
                .order_by(self.vector_table.columns.embeddings.l2_distance(embedding))
                .limit(limit)
            )
        elif algo == DBGlobal.COSINE_SIMILARITY:
            return self.search_session.execute(
                select(self.vector_table.columns.id)
                .order_by(
                    self.vector_table.columns.embeddings.cosine_distance(embedding)
                )
                .limit(limit)
            )
        elif algo == DBGlobal.MAX_INNER_PRODUCT:
            return self.search_session.execute(
                select(self.vector_table.columns.id)
                .order_by(
                    self.vector_table.columns.embeddings.max_inner_product(embedding)
                )
                .limit(limit)
            )
        return None

    def anninsert(self, embedding, table_name, insert_id=None):
        if insert_id != None:
            return self.search_session.execute(
                text(f"INSERT INTO {table_name} (id, embeddings) VALUES ({insert_id}, '{embedding}')")
            )

        return self.search_session.execute(
            text(f"INSERT INTO {table_name} (embeddings) VALUES ('{embedding}')")
        )

    def annupdate(self, id, embedding, table_name):
        return self.search_session.execute(
            text(f"UPDATE {table_name} SET embeddings='{embedding}' WHERE id={id}")
        )

    def anndelete(self, id, table_name):
        return self.search_session.execute(
            text(f"DELETE FROM {table_name} WHERE id={id}")
        )

    def annfilteredsearch(self, id, embedding, limit, algo):
        if algo == DBGlobal.L2_DISTANCE:
            return self.search_session.execute(
                select(self.vector_table.columns.id)
                .where(self.vector_table.columns.id < id)
                .order_by(self.vector_table.columns.embeddings.l2_distance(embedding))
                .limit(limit)
            )
        elif algo == DBGlobal.COSINE_SIMILARITY:
            return self.search_session.execute(
                select(self.vector_table.columns.id)
                .where(self.vector_table.columns.id < id)
                .order_by(
                    self.vector_table.columns.embeddings.cosine_distance(embedding)
                )
                .limit(limit)
            )
        elif algo == DBGlobal.MAX_INNER_PRODUCT:
            return self.search_session.execute(
                select(self.vector_table.columns.id)
                .where(self.vector_table.columns.id < id)
                .order_by(
                    self.vector_table.columns.embeddings.max_inner_product(embedding)
                )
                .limit(limit)
            )
        return None

    def anndatasetsize(self, table_name):
        return self.search_session.execute(text(f"SELECT COUNT(id) FROM {table_name}")).fetchall()[0][0]
    
    def anndatasetmaxid(self, table_name):
        return self.search_session.execute(text(f"SELECT MAX(id) FROM {table_name}")).fetchall()[0][0]

    def returned_rows(self, response):
        return response.fetchall()

    def get_by_id_batch(self, ids):
        vectors = []
        for id in ids:
            vectors.append(self.get_by_id(id))
        return vectors
