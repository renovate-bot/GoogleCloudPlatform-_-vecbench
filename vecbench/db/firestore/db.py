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
import os
import time
from db.dbglobal import DBGlobal
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.api_core.client_options import ClientOptions
import google.api_core.exceptions
import google.auth
from google.cloud import firestore
from google.cloud import firestore_admin_v1
from google.cloud.firestore_v1 import aggregation
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.bulk_writer import BulkWriter
from google.cloud.firestore_v1.vector import Vector
import metrics
import numpy as np

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()


class Firestore:
    _EMBEDDING_FIELD = "embedding"
    _DISTANCE_MAPPING = {
        DBGlobal.L2_DISTANCE: DistanceMeasure.EUCLIDEAN,
        DBGlobal.COSINE_SIMILARITY: DistanceMeasure.COSINE,
        DBGlobal.MAX_INNER_PRODUCT: DistanceMeasure.DOT_PRODUCT,
    }

    def __init__(self, config):
        self.type = "Firestore"
        api = config["public_endpoint_url"]
        self.project_id = config["project_id"]
        self.database_id = config["database_id"]

        # firebase_admin.initialize_app will fail if invoked more than once, but
        # we create an instance of this class for every worker, so we need to
        # defend against that here.
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
        options = ClientOptions(api_endpoint=api)
        self.client = firestore.Client(
            project=self.project_id,
            database=self.database_id,
            client_options=options,
        )
        self.admin_client = firestore_admin_v1.FirestoreAdminClient(
            client_options=options
        )
        metrics_type = metrics.NOOP_METRICS
        run_id = config["run_id"]
        self.metrics = metrics.get_metrics(metrics_type, run_id)
        self.table_exists = None

    def check_table(self, table_name):
        if self.table_exists is not None:
            return
        # 'Tables' in Firestore aren't created but exist by virtue of data being
        # created at a path, so we check the count of documents to see if it
        # exists
        # NOTE: This idea is a little finnicky, because the workers start up
        # sequentially and all do this check, so the later workers will see the
        # written documents of the earlier workers
        count_query = (
            self.client.collection(table_name).limit(50001).count(alias="all")
        )
        count = count_query.get()[0][0].value
        logging.info(f"Checking Kind Size:{table_name} with size {count}")
        if count >= 50000:
            self.table_exists = True

    def load_table(self, table_name):
        self.kind = table_name
        pass

    def create_index(self, table_name, vector_dimensions):
        index = firestore_admin_v1.Index(
            query_scope=firestore_admin_v1.Index.QueryScope.COLLECTION,
            fields=[
                firestore_admin_v1.Index.IndexField(
                    field_path=self._EMBEDDING_FIELD,
                    vector_config=firestore_admin_v1.Index.IndexField.VectorConfig(
                        dimension=vector_dimensions,
                        flat=firestore_admin_v1.Index.IndexField.VectorConfig.FlatIndex(),
                    ),
                )
            ],
        )
        create_index_request = firestore_admin_v1.CreateIndexRequest(
            parent=f"projects/{self.project_id}/databases/{self.database_id}/collectionGroups/{table_name}",
            index=index,
        )
        try:
            logging.info("Creating Index")
            operation = self.admin_client.create_index(
                request=create_index_request
            )
            logging.info("Waiting for index creation to complete...")
            response = operation.result()
            logging.info(response)
        except google.api_core.exceptions.AlreadyExists:
            logging.info("Index already exists")

    def CreateTable(self, table_name, db_recreate, vector_dimensions):
        self.create_index(table_name, vector_dimensions)

    def populate(self, table_name, data, start):
        pid = os.getpid()
        logging.info(f"Populating table:{table_name} with {len(data)}")
        if len(data[0]) == 2:
            self.populate_with_id(table_name, data)
        else:
            self.populate_without_id(table_name, data, start)

    def populate_with_id(self, table_name, data):
        bulk_writer = self.client.bulk_writer()
        coll_ref = self.client.collection(table_name)
        for i, (id, embedding) in enumerate(data):
            # Firestore uses string paths for its documents, so an ID of 100
            # would sort before an ID of 2. Since we want to find the MAX ID so
            # we know how many documents are in a dataset, we are using a field
            # to hold the 'id'
            bulk_writer.set(
                coll_ref.document(str(id)),
                {
                    self._EMBEDDING_FIELD: self.create_firestore_vector(
                        embedding
                    ),
                    "id": id,
                },
            )
        bulk_writer.flush()

    def populate_without_id(self, table_name, data, start_id):
        bulk_writer = self.client.bulk_writer()
        coll_ref = self.client.collection(table_name)
        for i, embedding in enumerate(data):
            id = start_id + i
            bulk_writer.set(
                coll_ref.document(str(id)),
                {
                    self._EMBEDDING_FIELD: self.create_firestore_vector(
                        embedding
                    ),
                    "id": id,
                },
            )
        bulk_writer.flush()

    def get_by_id(self, id):
        doc = self.client.collection(self.kind).document(id).get()
        response = [
            (id, np.array(doc.to_dict()[self._EMBEDDING_FIELD].value)),
        ]
        return response

    def get_by_id_batch(self, ids):
        vectors = []
        for id in ids:
            vectors.append(self.get_by_id(id))
        return vectors

    def configure_search_session(
        self, benchmark_config
    ):
        pass

    def set_value(self, table_name):
        pass

    def index_embeddings(self, benchmark_config, vector_table):
        pass

    def returned_rows(self, response):
        res = []
        for r in response:
            res.append([r.id])
        return res

    def anndatasetsize(self, table_name):
        count_query = (
            self.client.collection(table_name)
            .count(alias="all")
        )
        count = count_query.get()[0][0].value
        return count

    def anninsert(self, embedding, table_name, insert_id=None):
        if insert_id != None:
          self.client.collection(table_name).document(str(insert_id)).update({
              self._EMBEDDING_FIELD: self.create_firestore_vector(embedding),
              "id": insert_id,
          })
          return
        self.client.collection(table_name).document().set({
            self._EMBEDDING_FIELD: self.create_firestore_vector(embedding),
        })

    def annupdate(self, id, embedding, table_name):
        self.client.collection(table_name).document(str(id)).update({
            self._EMBEDDING_FIELD: self.create_firestore_vector(embedding),
        })

    def anndelete(self, id, table_name):
        self.client.collection(table_name).document(str(id)).delete()
        # Firestore's delete method only returns the timestamp that the request
        # was received by the server, and not whether the delete made any
        # changes. This is a hack to create a fake response that the system
        # expects to see.
        ret = lambda: None
        ret.rowcount = 1
        return ret

    def annsearch(self, embedding, limit, algo):
        distance_measure = self._DISTANCE_MAPPING[algo]
        results = (
            self.client.collection(self.kind)
            .select("__name__")
            .find_nearest(
                vector_field=self._EMBEDDING_FIELD,
                query_vector=self.create_firestore_vector(embedding),
                distance_measure=distance_measure,
                limit=limit,
            )
            .get()
        )
        return results

    def annfilteredsearch(self, id, embedding, limit, algo):
        pass

    def create_firestore_vector(self, embedding):
        float_list = [float(i) for i in embedding]
        return Vector(float_list)
