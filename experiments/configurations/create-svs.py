import json

threads = [1]
ws_constructs = [32]
ws_search = [230, 240, 250, 260, 270, 280, 290, 300]
graph_degree = [32]
qantization = ["8"]
for algo in ["svs"]:
    configs = []
    for thread in threads:
        for ws_construct in ws_constructs:
            for quant in qantization:
                for graph_d in graph_degree:
                    config = {
                        "name": f"svs-test-graph-{graph_d}-ws-con-{ws_construct}-quant-{quant}-threads-{thread}",
                        "engine": "redis",
                        "connection_params": {},
                        "collection_params": {
                            "algorithm": algo,
                            f"{algo}_config": {"NUM_THREADS": thread, "GRAPH_DEGREE": graph_d, "WS_CONSTRUCTION": ws_construct, "QUANTIZATION": quant},
                        },
                        "search_params": [],
                        "upload_params": {
                            "parallel": 128,
                            "algorithm": algo,
                        },
                }
                    for client in [1, 2, 4, 8]:
                        for ws_s in ws_search:
                            test_config = {
                                "algorithm": algo,
                                "parallel": client,
                                "search_params": {"WS_SEARCH": ws_s}
                            }
                            config["search_params"].append(test_config)
                    configs.append(config)
    fname = f"svs-test.json"
    with open(fname, "w") as json_fd:
        json.dump(configs, json_fd, indent=2)
        print(f"created {len(configs)} configs for {fname}.")
