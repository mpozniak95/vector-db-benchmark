import multiprocessing as mp
import time
import uuid
from typing import List, Optional

from opensearchpy import OpenSearch

from engine.base_client.upload import BaseUploader
from engine.clients.opensearch.config import OPENSEARCH_INDEX, get_opensearch_client


class ClosableOpenSearch(OpenSearch):
    def __del__(self):
        self.close()


class OpenSearchUploader(BaseUploader):
    client: OpenSearch = None
    upload_params = {}

    @classmethod
    def get_mp_start_method(cls):
        return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.client = get_opensearch_client(host, connection_params)
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        if metadata is None:
            metadata = [{}] * len(vectors)
        operations = []
        for idx, vector, payload in zip(ids, vectors, metadata):
            vector_id = uuid.UUID(int=idx).hex
            operations.append({"index": {"_id": vector_id}})
            if payload:
                operations.append({"vector": vector, **payload})
            else:
                operations.append({"vector": vector})

        cls.client.bulk(
            index=OPENSEARCH_INDEX,
            body=operations,
            params={
                "timeout": 300,
            },
        )

    @classmethod
    def post_upload(cls, _distance):
        cls.client.indices.forcemerge(
            index=OPENSEARCH_INDEX,
            max_num_segments=1,
            request_timeout=600,
        )
        cls.client.indices.refresh(index=OPENSEARCH_INDEX)
        
        # Wait for index to reach green status
        cls._wait_for_green_status()
        
        return {}
    
    @classmethod
    def _wait_for_green_status(cls, timeout=1200, check_interval=5):
        """Wait for the index to reach green status (all shards are active)"""
        print(f"Waiting for index '{OPENSEARCH_INDEX}' to reach green status...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                health = cls.client.cluster.health(
                    index=OPENSEARCH_INDEX,
                    wait_for_status="green",
                    timeout=5
                )
                
                status = health.get("status", "unknown")
                print(f"Index status: {status}")
                
                if status == "green":
                    print(f"Index '{OPENSEARCH_INDEX}' reached green status")
                    return
                    
            except Exception as e:
                print(f"Error checking cluster health: {e}")
            
            print(f"Index status not green yet, waiting {check_interval}s...")
            time.sleep(check_interval)
        
        print(f"Warning: Index '{OPENSEARCH_INDEX}' did not reach green status within {timeout}s timeout")
