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

import os
from collections.abc import Sequence
from typing import Dict, List, Union, Tuple, Iterable
from pinecone.grpc import PineconeGRPC as pc, GRPCIndex
from pinecone import ServerlessSpec, PineconeApiException, ForbiddenException, PodSpec, IndexList, NotFoundException, FetchResponse, PineconeException
import time
import logging
import numpy, numpy.typing
from types import SimpleNamespace
from sqlalchemy import true
from urllib3.exceptions import ProtocolError

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig()


def api_backoff(api_call):  # decorator to handle rate limiting issues

    def backoff_wrapper(*args, **kwargs):
        for attempt in range(40):
            sleep_secs = 0.1 * (attempt * attempt)
            try:
                if (attempt > 0):
                    time.sleep(sleep_secs)
                return api_call(*args, **kwargs)
            except (PineconeApiException, ForbiddenException) as e:
                # '429: too many requests' or '504: gateway timeout' errors are the most common,
                # and due to rate limiting and/or overloading.
                # Mysterious, isolated 403 errors also happen *sometimes* for no reason I have
                # been able to discern. Retrying for those so that they don't ruin benchmark results.
                if (e.status in (429, 504, 403)):
                    if (attempt > 38):
                        logging.info(f"Pinecone API backoff insufficient on "
                                     f"{os.getpid()}. Service overloaded?")
                        raise e
                    logging.info(f"Pinecone error encountered by {os.getpid()} : ({e.status}), waiting "
                                 f"{0.1 * ((attempt + 1) * (attempt + 1)) * 1000}"
                                 "ms before retry number "
                                 f"{attempt + 1}...")
                    continue
                else:
                    raise e
            except (ProtocolError) as e:
                # urllib errors. Probably caused by server overloading.
                logging.info(f"Protocol error while accessing Pinecone: {e}. Retrying with "
                             f"{0.1 * ((attempt + 1) * (attempt + 1)) * 1000} ms delay...")
            # Only the below should appear with PineconeGRPC
            except (PineconeException) as e:
                # Pinecone GRPC errors are terribly formatted, so we just retry on all of them.
                if (attempt > 38):
                    logging.info(f"Pinecone API backoff insufficient on "
                                 f"{os.getpid()}. Service overloaded?")
                    raise e
                logging.info(f"Pinecone error encountered by {os.getpid()} : ({e}), waiting "
                             f"{0.1 * ((attempt + 1) * (attempt + 1)) * 1000}"
                             "ms before retry number "
                             f"{attempt + 1}...")
                continue

    return backoff_wrapper


# Type aliases
# Return this to its rightful place once we upgrade to 3.11
# upsert_format: TypeAlias = Iterable[Dict[str, str | Iterable[float] | Dict[str, str]]]


class Pinecone:

    _ALGO_TO_INDEX_TYPE: Dict[str, str] = {
        "vector_cosine_ops": "cosine",
        "vector_l2_ops": "euclidean",
        "vector_ip_ops": "dotproduct"
    }
    SUPPORTED_ALGOS = _ALGO_TO_INDEX_TYPE.keys()

    def __init__(self, config):
        self.dbconfig = config
        self.pinecone = pc(config['api_key'])
        self.index_loaded = False
        self.reload_data = False

    def load_index(self, index_name: str, algo: str) -> GRPCIndex:
        """Load an index for querying and data loading.
        Note that Pinecone does not have separate concepts of 'tables' and 'indexes'
        Pinecone indexes have a single distance metric, so an algorithm must be specified 
        in addition to the table name. 
        Use get_pinecone_index_name to retrieve the canonical/normalized name of an index
        on Pinecone.
        
        Must call load_index or create_index before running a workload.
        """
        pc_index_name: str = Pinecone.get_pinecone_index_name(index_name, algo)
        index: GRPCIndex | None = self.pinecone.Index(pc_index_name)
        if index is None:
            # TODO: Replace this with an Odyssey-specific exception type.
            raise (NotFoundException(0, "Pinecone index failed to initialize."))
        # logging.info(index.describe_index_stats()) # Disabled due to rate limiting issues
        self.index: GRPCIndex = index
        self.loaded_index_name: str = pc_index_name
        self.index_loaded = True
        self.index_max_id = self._get_index_size()
        return index

    @api_backoff
    def _get_index_size(self) -> int:
        return self.index.describe_index_stats()['total_vector_count']

    def create_index(self, index_name: str, db_recreate: bool, vector_dimensions: int, benchmark_config):
        """Create a new index.
        Pinecone does not separate tables and indexes. As such, a distance metric
        algorithm must be specified at the time of index/table creation.
        """
        algo = benchmark_config["config"]["algo"]
        index_type = self._ALGO_TO_INDEX_TYPE[algo]
        pc_index_name = Pinecone.get_pinecone_index_name(index_name, algo)
        index_exists = self._index_exists(pc_index_name)
        if (db_recreate or not index_exists):
            if index_exists:
                logging.info(f'Deleting existing index {pc_index_name}...')
                self.pinecone.delete_index(pc_index_name)
            if (self.dbconfig['serverless']):
                self._create_serverless_index(pc_index_name, vector_dimensions, index_type)
            else:
                self._create_pod_index(pc_index_name, vector_dimensions, index_type)
        return self.load_index(index_name, algo)

    def _create_serverless_index(self, pc_index_name: str, vector_dimensions: int, index_type: str):
        logging.info(f'Creating serverless index {pc_index_name}...')
        self.pinecone.create_index(name=pc_index_name,
                                   dimension=vector_dimensions,
                                   metric=index_type,
                                   spec=ServerlessSpec(cloud=self.dbconfig['cloud'], region=self.dbconfig['region']))

    def _create_pod_index(self, pc_index_name: str, vector_dimensions: int, index_type: str):
        logging.info(f'Creating index {pc_index_name} of type {self.dbconfig["pod_type"]} '
                     f'with {self.dbconfig["replicas"]} replicas and {self.dbconfig["shards"]} shards...')
        self.pinecone.create_index(name=pc_index_name,
                                   dimension=vector_dimensions,
                                   metric=index_type,
                                   spec=PodSpec(environment=self.dbconfig['environment'],
                                                pod_type=self.dbconfig['pod_type'],
                                                replicas=self.dbconfig['replicas'],
                                                shards=self.dbconfig['shards']))
        time.sleep(10)  # Give Pinecone some time to finish provisioning

    def populate(self, table_name, data, start, algo):
        if not self.index_loaded or (not Pinecone.get_pinecone_index_name(table_name, algo) == self.loaded_index_name):
            self.load_index(table_name, algo)
        logging.info(f"Populating index:{self.loaded_index_name} with {len(data)} vectors")
        upsert_data: Iterable[Dict[str, Union[str, Iterable[float], Dict[str,str]]]] = []
        # Pinecone does not support integer vectors -- convert all values to float before insert
        # orig_id included in metadata to enable filtering
        if len(data[0]) == 2:  # labeled data
            upsert_data = ({
                "id": str(x),
                "values": [float(s) for s in y],
                "metadata": {
                    "orig_id": str(x)
                }
            } for (x, y) in data)
        else:  # unlabeled data, generate labels
            upsert_data = ({
                "id": str(i + start),
                "values": (float(s) for s in data[i]),
                "metadata": {
                    "orig_id": str(i + start)
                }
            } for i in range(0, len(data)))
        self.populate_with_id(upsert_data)

    def populate_with_id(self, data: Iterable[Dict[str, Union[str, Iterable[float], Dict[str,str]]]]) -> None:
        # Pinecone recommends a maximum batch size of 100 vectors per upsert request
        # NB: 2MB absolute limit -- could 100 wide vectors hit this?
        # May need to make this configurable in future.
        stream = iter(data)
        empty = False
        while not empty:
            batch = []
            try:
                for _ in range(0, 100):
                    batch.append(next(stream))
            except StopIteration:
                empty = True
            finally:
                if batch:
                    self._upsert_vecs(batch, False)

    @api_backoff
    def _upsert_vecs(self, data: Iterable[Dict[str, Union[str, Iterable[float], Dict[str,str]]]], is_async: bool) -> None:
        """upsert data format:
        [{"id": str(data_id), "values": list(float), "metadata":{"metadata_tag":str(tag_value)}}]
        """
        # Despite the name of the parameter, 'async_req=False' does not make operatious synchronous,
        # it just makes the _request_ return synchronously. Index updates may still take hours.
        self.index.upsert(vectors=list(data), async_req=is_async)

    def index_embeddings(self, benchmark_config, vector_table) -> None:
        pass  # no-op, Pinecone does not support unindexed tables.

    def configure_search_session(self, benchmark_config) -> None:
        pass  # no-op, Pinecone's proprietary index algorithm is not user configurable.

    def set_value(self, table_name) -> None:
        pass  # no-op, Pinecone does not have sequence objects.

    @api_backoff
    def annsearch(self, embedding, limit, algo):
        return self.index.query(vector=embedding.tolist(), top_k=limit)

    @api_backoff
    def anninsert(self, embedding, table_name, insert_id) -> None:
        self._upsert_vecs([{
            "id": str(insert_id),
            "values": [float(n) for n in embedding],
            "metadata": {
                "orig_id": str(insert_id)
            }
        }], False)

    @api_backoff
    def annupdate(self, id, embedding, table_name) -> None:
        self._upsert_vecs([{
            "id": str(id),
            "values": [float(n) for n in embedding],
            "metadata": {
                "orig_id": str(id)
            }
        }], False)

    @api_backoff
    def anndelete(self, id, table_name) -> SimpleNamespace:
        self.index.delete([str(id)], async_req=False)

        # Goofy hack to make workloads run until we have a better interface.
        # Pinecone deletes always succeed, even if the id doesn't exist in the index.
        return SimpleNamespace(rowcount=1)

    def annfilteredsearch(self, id, embedding, limit, algo) -> None:
        # Pinecone supports filtering on metadata, but not on IDs
        # See https://docs.pinecone.io/docs/metadata-filtering
        # Not Yet Implemented
        pass

    def anndatasetsize(self, table_name) -> int:
        return self.index.describe_index_stats()['total_vector_count']

    @api_backoff
    def get_by_id(self, id) -> List[Tuple[int, numpy.typing.NDArray]]:
        return [(id, numpy.asarray(self.index.fetch(ids=[str(id)])['vectors'][str(id)]['values']))]

    def _index_exists(self, index_name) -> bool:
        index_response: IndexList = self.pinecone.list_indexes()
        for name in index_response.names():
            if name == index_name:
                return True
        return False

    # NB / TODO: We should probably regularize results from db drivers so
    # that we're not relying on this weird pattern
    def returned_rows(self, response) -> List[Tuple[int]]:
        # Pinecone also returns a 'score' with each result
        return [(int(r['id']),) for r in response['matches']]

    @staticmethod
    def get_pinecone_index_name(index_name: str, algo: str) -> str:
        return f'{index_name.replace("_","-")}-{Pinecone._ALGO_TO_INDEX_TYPE[algo]}'

    def get_by_id_batch(self, ids) -> List[List[Tuple[int, numpy.typing.NDArray]]]:
        searchVectors: List[str] = [str(id) for id in ids]
        fetchedVectors: FetchResponse = self.index.fetch(ids=searchVectors)
        vectors: List[List[Tuple[int, numpy.typing.NDArray]]] = []
        for id in ids:
            try:
                vectors.append([(id, numpy.asarray(fetchedVectors['vectors'][str(id)]['values']))])
            except KeyError as e:
                logging.info(f"Deleted vector {id} excluded from recall calculations.")
                continue
        return vectors
