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
from metrics.metrics import Metrics
import time
import logging
import os
from db.dbglobal import DBGlobal
import mysql.connector
import numpy as np
import metrics


logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()

class CsqlMySQL:
    def __init__(self, config):
        self.type = "MySQL"
        self.table_exists = False
        self.metrics = metrics.get_metrics(metrics.NOOP_METRICS, config["run_id"])
        self.db = mysql.connector.connect(
                host=f"{config['ip']}",
                user=f"{config['user']}",
                password=f"{config['password']}",
                database=f"{config['database']}",
                autocommit=True)
        self.database_name = config['database']
        self.index_name = 'index_vec'
        
        self.num_leaves_to_search = 0

    def load_table(self, table_name):
        self.vector_table = table_name #vector_table
        return table_name #pass

    def CreateTable(self, table_name, db_recreate, vector_dimensions):
        cursor = self.db.cursor()
        sql = f"DROP TABLE IF EXISTS {table_name}"
        cursor.execute(sql)
        sql = f"Create table if not exists {table_name} (id int primary key, embeddings vector({vector_dimensions}) using varbinary)"
        cursor.execute(sql)
        cursor.close()

    def populate(self, table_name, data, start):
        pid = os.getpid()
        self.dataset_name = table_name
        cursor = self.db.cursor()
        tags = {
                "tool": "db.populate",
                "type": self.type,
                "pid": pid,
                "dataset_name": table_name,
            }
        if self.table_exists == False:
            if len(data[0]) == 2:
                self.populate_with_id(cursor, table_name, data, tags)
            else:
                self.populate_without_id(cursor, table_name, data, start, tags)
        cursor.close()

    def populate_with_id(self, cursor, table_name, data, tags):
        for i, (id, embedding) in enumerate(data):
            start = time.time()
            cursor.execute(
                f"INSERT INTO {table_name} (id, embeddings) VALUES (%s, string_to_vector(%s))",
                (id, np.array_str(embedding),),
            )
            end = time.time()
            self.metrics.collect("insert", tags, "elapsed", (end - start))
            self.metrics.collect("insert", tags, "inserted", i)

    def populate_without_id(self, cursor, table_name, data, start_id, tags):
        for i, embedding in enumerate(data):
            embedding_str = f"[{' '.join(str(dim) for dim in embedding)}]"
            start = time.time()
            cursor.execute(
                f"INSERT INTO {table_name} (id, embeddings) VALUES (%s, string_to_vector('{embedding_str}'))",
                (start_id + i,),
            )
            end = time.time()
            self.metrics.collect("insert", tags, "elapsed", (end - start))
            self.metrics.collect("insert", tags, "inserted", i)

    def index_embeddings(self, benchmark_config, vector_table):
        index_recreate=benchmark_config["config"]["index_recreate"]
        index_type=benchmark_config["config"]["index_type"]
        index_config=benchmark_config["config"]["index_config"]
        algo=benchmark_config["config"]["algo"]
        column_name = 'embeddings'
        num_partitions = index_config[0]['num_leaves']

        self.num_leaves_to_search = benchmark_config["config"]["num_leaves_to_search"]
        if index_recreate == True:
            logging.info(
                f"Indexing table type:{index_type} with algo: {algo} config: {index_config}"
            )
            cursor = self.db.cursor()
            
            try:
                cursor.callproc("mysql.drop_vector_index",[f'{self.database_name}.{vector_table}_{self.index_name}'])
                logging.info("Dropped index")
            except mysql.connector.Error as err:
                logging.info(f"Error dropping index: {err}")
            finally:
                cursor.close()
            
            distance = ""
            if "vector_cosine_ops" in algo:
                distance = "COSINE"
            if "vector_l2_ops" in algo:
                distance = "SQUARED_L2"
            expected_sample_size = self.anndatasetsize(vector_table)
            cursor = self.db.cursor()           
            try:
                if num_partitions == 0:
                    cursor.callproc("mysql.create_vector_index", [f"{vector_table}_{self.index_name}", f"{self.database_name}.{vector_table}", f"{column_name}", f"index_type={index_type}, table_size={expected_sample_size}, distance_measure={distance}"])
                else:
                    cursor.callproc("mysql.create_vector_index", [f"{vector_table}_{self.index_name}", f"{self.database_name}.{vector_table}", f"{column_name}", f"index_type={index_type}, table_size={expected_sample_size}, distance_measure={distance}, num_partitions={num_partitions}"])
                logging.info("Created index index_vec on table {vector_table}")
            except mysql.connector.Error as err:
                print(f"Error creating index: {err}")
            finally:
                cursor.close()

    def configure_search_session(self, benchmark_config):
        if num_leaves_to_search not in benchmark_config:
            self.num_leaves_to_search = 0 
        else:
            self.num_leaves_to_search = int(benchmark_config['num_leaves_to_search'])

    def set_value(self, table_name):
        pass

    def get_by_id(self, id):
        sql = f"SELECT id, VECTOR_TO_STRING(embeddings) FROM {self.vector_table} WHERE id = %s"
        cursor = self.db.cursor()
        cursor.execute(sql, (id,))
        result = cursor.fetchall()
        new_data = []
        for item in result:
            id_value, list_str = item
            float_list = [float(num) for num in list_str.strip('[]').split(',')]
            array = np.array(float_list, dtype=np.float32)
            new_data.append((id_value, array))
        return new_data

    def annsearch(self, embedding, limit, algo):
        sql = ""
        if self.num_leaves_to_search > 0:
            sql = f"SELECT id FROM {self.vector_table} WHERE NEAREST (EMBEDDINGS) TO (STRING_TO_VECTOR('{embedding}'), 'NUM_NEIGHBORS = {limit}, NUM_PARTITIONS = {self.num_leaves_to_search}')"
        else:
            sql = f"SELECT id FROM {self.vector_table} WHERE NEAREST (EMBEDDINGS) TO (STRING_TO_VECTOR('{embedding}'), 'NUM_NEIGHBORS = {limit}')"
        cursor = self.db.cursor()
        cursor.execute(sql)
        return cursor

    def anninsert(self, embedding, table_name):
        cursor = self.db.cursor()
        cursor.execute(
                f"INSERT INTO {table_name} (embeddings) VALUES (string_to_vector(%s))",
                (np.array_str(embedding),),
            )
        cursor.close() 

    def annupdate(self, id, embedding, table_name):
        cursor = self.db.cursor()
        cursor.execute(
                f"UPDATE {table_name} SET embeddings=string_to_vector(%s) WHERE id={id}",
                (np.array_str(embedding),),
            )
        cursor.close() 

    def anndelete(self, id, table_name):
        cursor = self.db.cursor()
        cursor.execute(f"DELETE FROM {table_name} WHERE id={id}")
        cursor.close() 

    def annfilteredsearch(self, id, embedding, limit, algo):
        sql = ""
        if self.num_leaves_to_search > 0:
            sql = f"SELECT id FROM {self.vector_table} WHERE NEAREST (EMBEDDINGS) TO (STRING_TO_VECTOR('{embedding}'), 'NUM_NEIGHBORS = {limit}, NUM_PARTITIONS = {self.num_leaves_to_search}') AND id < {id}"
        else:
            sql = f"SELECT id FROM {self.vector_table} WHERE NEAREST (EMBEDDINGS) TO (STRING_TO_VECTOR('{embedding}'), 'NUM_NEIGHBORS = {limit}') AND id < {id}"
        cursor = self.db.cursor()
        cursor.execute(sql)
        return cursor

    def anndatasetsize(self, table_name):
        cursor = self.db.cursor()
        sql = f"select COUNT(1) FROM {table_name}"
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        return results[0][0] 

    def returned_rows(self, response):
        return response.fetchall()
    
    def get_by_id_batch(self, ids):
        vectors = []
        for id in ids:
            vectors.append(self.get_by_id(id))
        return vectors
