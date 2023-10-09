import os
import sys
import time
import json
import signal
import random
import socket
import subprocess
from functools import partial
from pathlib import Path
from threading import Thread
from metaflow.exception import MetaflowException
from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.plugins.parallel_decorator import ParallelDecorator, _local_multinode_control_task_step_func

class TensorFlowParallelDecorator(ParallelDecorator):

    name = "tensorflow"
    defaults = {}
    IS_PARALLEL = True

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):

        def _empty_mapper_task():
            pass

        if os.environ.get("METAFLOW_RUNTIME_ENVIRONMENT", "local") == "local":
            if ubf_context == UBF_CONTROL:
                env_to_use = getattr(self.environment, "base_env", self.environment)
                return partial(
                    _local_multinode_control_task_step_func,
                    flow,
                    env_to_use, 
                    step_func,
                    retry_count,
                )    
            return partial(_empty_mapper_task)

        return super().task_decorate(
            step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
        )

    def setup_distributed_env(self, flow):
        setup_tf_distributed(flow)

def setup_tf_distributed(run):

    from metaflow import current, S3

    num_nodes = current.parallel.num_nodes
    node_index = current.parallel.node_index

    local_ip = socket.gethostbyname_ex(socket.gethostname())[-1][0]
    s3 = S3(run=run)

    # Create a port id based on run id and node index to avoid clashes if the
    # workers run on same machine
    my_port = 40000 + (int(current.run_id) % 100) * 100 + node_index
    info_dict = {"node": node_index, "address": "{}:{}".format(local_ip, my_port)}
    key = os.path.join("tf_nodes", "node_{}.json".format(node_index))
    s3.put(key, json.dumps(info_dict))

    # Then poll for others
    all_workers = {node_index: info_dict}
    while len(all_workers) < num_nodes:
        print("Waiting for all worker tasks to register...")
        for other_node in range(num_nodes):
            if other_node not in all_workers:
                node_key = os.path.join("tf_nodes", "node_{}.json".format(other_node))
                node_info = s3.get(node_key, return_missing=True)  # use get_many
                if node_info.exists:
                    all_workers[other_node] = json.loads(node_info.blob)

        time.sleep(4.0 + random.random() * 3.0)

    my_task = {"type": "worker", "index": node_index}
    cluster = {
        "worker": [
            all_workers[node_id]["address"] 
            for node_id in range(num_nodes)
        ]
    }
    json_config = json.dumps({"cluster": cluster, "task": my_task})
    os.environ["TF_CONFIG"] = json_config

STEP_DECORATORS_DESC = [("tensorflow", ".TensorFlowParallelDecorator")]