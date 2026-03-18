from datetime import datetime
import os
import time
# import re
import math
import heapq
import random
# import faiss
# from typing import List, Optional
import numpy as np
# import plotly.graph_objects as go
# from scipy.io import loadmat
import threading   # added for thread safety in HNSW csv
import argparse 
import sys

# take num_index_per_subject as argument
parser = argparse.ArgumentParser()
parser.add_argument("--e", type=int, default=3, help="Number of index vectors to use per subject (default: 3)")
parser.add_argument("--M", type=int, default=32, help="HNSW parameter M (default: 32)")
parser.add_argument("--efc", type=int, default=300, help="HNSW parameter ef_construction (default: 300)")

parser.add_argument("--efs_start", type=int, default=100, help="HNSW parameter ef_search (default: 100)")
parser.add_argument("--efs_end", type=int, default=1000, help="HNSW parameter ef_search (default: 1000)")
parser.add_argument("--efs_step", type=int, default=100, help="HNSW parameter ef_search step (default: 100)")
# parser.add_argument("--efs", type=int, default=400, help="HNSW parameter ef_search (default: 400)")
# add argument for input dataset folder
parser.add_argument("--root_folder", type=str, default=f"/home/nishkal/sg/iris_indexing/datasets/iris_syn", help="Root folder of the dataset (default: /home/nishkal/sg/iris_indexing/datasets/iris_syn)")
# add an out file argument to save results
# parser.add_argument("--out", type=str, default=f"HNSW_syn_e3_M32_efc300_efs400.txt", help="Output file to save results (default: results.txt)")
# parser.add_argument("--out_dir", type=str, default=f"results/hnsw/", help="Directory to save results (default: results/hnsw/)")
parser.add_argument("--out_csv", type=str, default=f"HNSW_syn_results", help="Results CSV File Name without the extension (default: HNSW_syn_results)")
args = parser.parse_args()

from pathlib import Path as P

# variable that holds this
num_index_per_subject = args.e
M = args.M
ef_construction = args.efc

# ef_search = args.efs


out_dir = P(f"results/hnsw/")

out_dir.mkdir(parents=True, exist_ok=True)
# out_file = f"{out_dir}/HNSW_syn_e{num_index_per_subject}_M{M}_efc{ef_construction}_efs{ef_search}.txt"


root_folder = args.root_folder

# log all the arguments to the output file

import logging as lg
# log the results into a csv file with columns: num_index_per_subject, M, ef_construction, ef_search, hit_rate, avg_time_ms, total_queries
# pipe output to the terminal as well as file
# python HNSW_syn_e_M_efc_efs.py --e 3 --M 32 --efc 300 --efs 400
lg.basicConfig(
    level=lg.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # lg.FileHandler(out_file),
        lg.StreamHandler()
    ]
)

csv_lock = threading.Lock()  # added

out_csv:P = out_dir.parent/f"{args.out_csv.replace(".csv","")}.csv"
import pandas as pd
def get_results_set(out_:P) -> set:
    if out_.exists():
        return set(map(tuple,pd.read_csv(out_)[['e','M','efc','efs']].values))
    else:
        return set()
        
# results_df[].to_numpy()


if not out_csv.exists():
    with csv_lock, open(out_csv, "w") as f:
        f.write("date,time,e,M,efc,efs,index_build_time,hit_rate,avg_time_ms,total_queries\n")
    # csv_lock.release()  
else:
    # check if all the required efs already exists, and break
    # pass
    all_efs_done = True
    r_set=get_results_set(out_csv)
    for ef_search in range(args.efs_start, args.efs_end + 1, args.efs_step):
        if (num_index_per_subject,M,ef_construction,ef_search) not in r_set:
            all_efs_done = False
            break
    if all_efs_done:
        lg.info(f"ALL EXPERIMENTS WITH PARAMS {(num_index_per_subject,M,ef_construction)=} ALREADY DONE --- EXIING")
        sys.exit(0)

class Node:
    def __init__(self, idx: int, vector: np.ndarray, level: int):
        self.idx = idx
        self.vector = vector
        self.level = level
        self.neighbors = [[] for _ in range(level + 1)]
        self.deleted = False


class HNSW:
    def __init__(self, M=16, ef_construction=200, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.M = M
        self.ef_construction = ef_construction
        self.level_mult = 1.0 / math.log(M) if M > 1 else 1.0
        self.entry_point = None
        self.max_level = -1
        self.nodes = []
        self._visit_mark = 1
        self._visited = {}

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(a - b))

    def _get_random_level(self):
        r = random.random()
        if r == 0.0:
            r = 1e-16
        return int(math.floor(-math.log(r) * self.level_mult))

    def _next_visit(self):
        self._visit_mark += 1
        if self._visit_mark % 1000000 == 0:
            self._visited.clear()

    def _is_visited(self, idx):
        return self._visited.get(idx, 0) == self._visit_mark

    def _set_visited(self, idx):
        self._visited[idx] = self._visit_mark

    def _greedy_search_layer(self, query_vec, entry_idx, level):
        current = entry_idx
        current_dist = self._distance(query_vec, self.nodes[current].vector)
        improved = True
        while improved:
            improved = False
            for nb in self.nodes[current].neighbors[level]:
                d = self._distance(query_vec, self.nodes[nb].vector)
                if d < current_dist:
                    current = nb
                    current_dist = d
                    improved = True
        return current

    def _search_layer(self, query_vec, entry_idx, ef, level):
        candidates = []
        result_heap = []

        self._next_visit()
        entry_dist = self._distance(query_vec, self.nodes[entry_idx].vector)
        heapq.heappush(candidates, (entry_dist, entry_idx))
        heapq.heappush(result_heap, (-entry_dist, entry_idx))
        self._set_visited(entry_idx)

        while candidates:
            top_dist, top_idx = candidates[0]
            worst_dist = -result_heap[0][0]
            if top_dist > worst_dist and len(result_heap) >= ef:
                break
            heapq.heappop(candidates)
            for nb in self.nodes[top_idx].neighbors[level]:
                if self._is_visited(nb):
                    continue
                self._set_visited(nb)
                d = self._distance(query_vec, self.nodes[nb].vector)
                if len(result_heap) < ef or d < -result_heap[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(result_heap, (-d, nb))
                    if len(result_heap) > ef:
                        heapq.heappop(result_heap)

        res = [(-dist, idx) for (dist, idx) in result_heap]
        res.sort(key=lambda x: x[0])
        return res

    def _select_neighbors(self, query_vec, candidates, M):
        selected = []
        for dist_q, cand_idx in candidates:
            cand_vec = self.nodes[cand_idx].vector
            ok = True
            for sel_idx in selected:
                if self._distance(cand_vec, self.nodes[sel_idx].vector) < dist_q:
                    ok = False
                    break
            if ok:
                selected.append(cand_idx)
            if len(selected) >= M:
                break
        return selected

    def _select_neighbors_from_ids(self, query_vec, ids, M):
        cand_with_dist = [(self._distance(query_vec, self.nodes[idx].vector), idx) for idx in ids]
        cand_with_dist.sort(key=lambda x: x[0])
        return self._select_neighbors(query_vec, cand_with_dist, M)

    def add_item(self, vector, idx=None):
        if idx is None:
            idx = len(self.nodes)
        vector = np.asarray(vector, dtype=float)
        level = self._get_random_level()
        node = Node(idx, vector, level)
        if idx >= len(self.nodes):
            self.nodes.extend([None] * (idx - len(self.nodes) + 1))
        self.nodes[idx] = node

        if self.entry_point is None:
            self.entry_point = idx
            self.max_level = level
            return

        ep = self.entry_point
        for L in range(self.max_level, level, -1):
            ep = self._greedy_search_layer(vector, ep, L)

        for L in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, L)
            selected = self._select_neighbors(vector, candidates, self.M)
            node.neighbors[L] = selected.copy()

            for nb in selected:
                nb_neighbors = self.nodes[nb].neighbors[L]
                nb_neighbors.append(idx)
                if len(nb_neighbors) > self.M:
                    pruned = self._select_neighbors_from_ids(self.nodes[nb].vector, nb_neighbors, self.M)
                    self.nodes[nb].neighbors[L] = pruned

            if candidates:
                ep = candidates[0][1]

        if level > self.max_level:
            self.entry_point = idx
            self.max_level = level

    def search_knn(self, query_vec, k=1, ef_search=None):
        if self.entry_point is None:
            return []
        if ef_search is None:
            ef_search = max(k, 50)

        ep = self.entry_point
        for L in range(self.max_level, 0, -1):
            ep = self._greedy_search_layer(query_vec, ep, L)
        results = self._search_layer(query_vec, ep, ef_search, 0)
        return results[:k]



import time
import numpy as np
from typing import Tuple, List, Dict

def evaluate_top1_timing(hnsw_index,
                         index_labels: np.ndarray,
                         query_vectors: np.ndarray,
                         query_labels: np.ndarray,
                         ef_search: int = 20) -> Dict[str, object]:
    """
    For each query vector:
      - perform a top-1 search (k=1)
      - measure the time taken for the search call
      - check if the retrieved top-1 label matches the query label

    Returns a dict containing:
      - 'hit_rate': fraction of queries whose top-1 label matched
      - 'avg_time_s': average search time per query in seconds
      - 'times_s': list of per-query times (seconds) in the same order as query_vectors
      - 'total_queries': number of queries
      - 'successful_searches': number of searches that returned at least one neighbor
    """
    if len(index_labels) == 0 or len(query_labels) == 0:
        raise ValueError("Index or query set empty.")

    if len(query_vectors) != len(query_labels):
        raise ValueError("query_vectors and query_labels must have the same length.")

    times: List[float] = []
    hits = 0
    successful_searches = 0
    total_queries = len(query_labels)

    for qvec, qlabel in zip(query_vectors, query_labels):
        t0 = time.perf_counter()
        res = hnsw_index.search_knn(qvec, k=1, ef_search=ef_search)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times.append(elapsed)

        # If the index returned at least one neighbor, check label
        if len(res) > 0:
            successful_searches += 1
            # assume res is iterable of (distance, index) or similar
            # take the index of the first (top-1) result
            top_idx = res[0][1]
            if index_labels[top_idx] == qlabel:
                hits += 1

    hit_rate = hits / total_queries
    avg_time_s = float(sum(times) / total_queries) if total_queries > 0 else 0.0

    return {
        "hit_rate": hit_rate,
        "avg_time_s": avg_time_s,
        "times_s": times,
        "total_queries": total_queries,
        "successful_searches": successful_searches
    }


import os
import random
import numpy as np


def load_iris_txt_templates(root_folder, seed=42, pick='random', normalize=True,
                            num_index_per_subject=3):
    """
    Dataset structure:
    root_folder/
        9978/
            1_template.txt
            1_mask.txt
            ...
            10_template.txt
            10_mask.txt

    For each subject:
    - pick 3 pairs for indexing
    - remaining pairs go to query pool
    - finally sample 10000 queries randomly from the full query pool

    Assumption:
    - template.txt contains binary iris code bits
    - mask.txt contains binary validity bits
    - mask == 1 means valid
    - mask == 0 means invalid
    - final vector = template & mask
    """

    random.seed(seed)
    np.random.seed(seed)

    if not os.path.isdir(root_folder):
        raise ValueError(f"Root folder not found: {root_folder}")

    subject_folders = sorted([
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ])

    if not subject_folders:
        raise ValueError(f"No subject folders found in {root_folder}")

    def read_txt_bits(path):
        with open(path, "r") as f:
            content = f.read()

        bits = [int(ch) for ch in content if ch in "01"]

        if len(bits) == 0:
            raise ValueError(f"No binary digits found in {path}")

        return np.array(bits, dtype=np.uint8)

    def pair_to_vector(template_path, mask_path):
        template = read_txt_bits(template_path)
        mask = read_txt_bits(mask_path)

        if template.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"template: {template.shape} from {template_path}\n"
                f"mask: {mask.shape} from {mask_path}"
            )

        vec = (template & mask).astype(np.float32)

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    index_vectors = []
    index_labels = []
    query_pool_vectors = []
    query_pool_labels = []

    for subject in subject_folders:
        subject_path = os.path.join(root_folder, subject)

        pairs = []
        for i in range(1, 11):
            template_file = os.path.join(subject_path, f"{i}_template.txt")
            mask_file = os.path.join(subject_path, f"{i}_mask.txt")

            if os.path.exists(template_file) and os.path.exists(mask_file):
                pairs.append((i, template_file, mask_file))

        if len(pairs) == 0:
            lg.info(f"⚠️ No valid txt pairs found for subject {subject}, skipping.")
            continue

        if len(pairs) < num_index_per_subject:
            lg.info(f"⚠️ Subject {subject} has only {len(pairs)} pairs, skipping.")
            continue

        if pick == 'random':
            random.shuffle(pairs)
        elif pick == 'first':
            pairs = sorted(pairs, key=lambda x: x[0])
        else:
            raise ValueError("pick must be 'random' or 'first'")

        index_pairs = pairs[:num_index_per_subject]
        query_pairs = pairs[num_index_per_subject:]

        index_count = 0
        query_count = 0

        for idx_num, idx_template, idx_mask in index_pairs:
            try:
                idx_vec = pair_to_vector(idx_template, idx_mask)
                index_vectors.append(idx_vec)
                index_labels.append(subject)
                index_count += 1
            except Exception as e:
                lg.info(f"⚠️ Failed index pair for subject {subject}, image {idx_num}: {e}")

        for q_num, q_template, q_mask in query_pairs:
            try:
                q_vec = pair_to_vector(q_template, q_mask)
                query_pool_vectors.append(q_vec)
                query_pool_labels.append(subject)
                query_count += 1
            except Exception as e:
                lg.info(f"⚠️ Failed query pair for subject {subject}, image {q_num}: {e}")

        # lg.info(f"{subject}: index={index_count}, query_pool={query_count}")

    if len(index_vectors) == 0:
        raise ValueError("No index vectors created. Check txt file contents.")

    if len(query_pool_vectors) == 0:
        raise ValueError("No query pool vectors created.")

    # Convert index data
    index_vectors = np.vstack(index_vectors).astype(np.float32)
    index_labels = np.array(index_labels)

    # Randomly sample final queries from the full query pool
    # total_query_pool = len(query_pool_vectors)
    # if num_queries_to_sample > total_query_pool:
    #     lg.info(f"⚠️ Requested {num_queries_to_sample} queries, but only {total_query_pool} available.")
    #     num_queries_to_sample = total_query_pool

    # sampled_indices = np.random.choice(total_query_pool, size=num_queries_to_sample, replace=False)

    # query_pool_vectors = np.vstack(query_pool_vectors).astype(np.float32)
    # query_pool_labels = np.array(query_pool_labels)

    query_vectors = query_pool_vectors
    query_labels = query_pool_labels

    return index_vectors, index_labels, query_vectors, query_labels


# root_folder = "/home/nishkal/sg/iris_indexing/datasets/iris_syn"

index_vectors, index_labels, query_vectors, query_labels = load_iris_txt_templates(
    root_folder=root_folder,
    seed=42,
    pick='random',
    normalize=True,
    num_index_per_subject=num_index_per_subject,
)



hnsw = HNSW(M=M, ef_construction=ef_construction, random_seed=42)

st = time.perf_counter()


for i, vec in enumerate(index_vectors):
        hnsw.add_item(vec, idx=i)

end = time.perf_counter()
lg.info(f"Index construction time: {end - st:.4f} seconds")

lg.info(f"Index built with {len(index_vectors)} vectors.")




for ef_search in range(args.efs_start, args.efs_end + 1, args.efs_step):
    r_set = get_results_set(out_csv)
    if (num_index_per_subject,M,ef_construction,ef_search) in r_set:
        lg.info(f'Experiment with params {(num_index_per_subject,M,ef_construction,ef_search)=} already done')
        continue
    results = evaluate_top1_timing(hnsw,index_labels,query_vectors,query_labels,ef_search=ef_search)
    lg.info(f"Hit rate: {results['hit_rate']}")
    lg.info(f"Average search time (ms): {results['avg_time_s'] * 1000}")
    lg.info(f"Total queries: {results['total_queries']}")
    # save results to csv
    with csv_lock, open(out_csv, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d,%H:%M:%S')},{num_index_per_subject},{M},{ef_construction},{ef_search},{end - st:.4f},{results['hit_rate']:.04f},{results['avg_time_s'] * 1000:.4f},{results['total_queries']}\n")