# %%
import os
import time
import re
import math
import heapq
import random
import faiss
from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat

# %%
class Node:
    def __init__(self, idx: int, vector: np.ndarray, level: int):
        self.idx = idx
        self.vector = vector
        self.level = level
        self.neighbors = [[] for _ in range(level + 1)]
        self.deleted = False


class HNSWPL:
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
        alpha = 0.5
        level = int((r ** (-alpha) - 1) * self.level_mult)
        return level


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


# %%
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


# %%

def load_IrisLamp_templates(root_folder, seed=42, pick='random'):
    """
    Process a flat folder of template files named like: 001_L_01_template.npz
    Groups by subject-eye (e.g. '001_L'), picks exactly one file per group for indexing,
    and the rest for querying.

    Args:
        root_folder (str): path to folder containing .npz / .npy template files.
        seed (int): deterministic seed for random picking.
        pick (str): 'random' (default) or 'first' - selection strategy for the 1 index file.

    Returns:
        index_vectors: np.ndarray (M, D) float32
        index_labels:  np.ndarray (M,) strings like '001_L'
        query_vectors: np.ndarray (Q, D) float32 (empty if none)
        query_labels:  np.ndarray (Q,) strings
    """
    random.seed(seed)
    np.random.seed(seed)

    # regex to extract subject and eye from filename; flexible to variations
    # examples matched: 001_L_01_template.npz, 23_R_3_template.npz, 0001_L_12_template.npz
    pattern = re.compile(r'^(\d+)_([LRlr])_.*template', re.IGNORECASE)

    # collect files
    files = [f for f in os.listdir(root_folder) if f.lower().endswith(('.npz', '.npy'))]
    if not files:
        raise ValueError(f"No .npz/.npy files found in {root_folder}")

    groups = {}  # key -> list of filenames
    for f in sorted(files):
        m = pattern.match(f)
        if not m:
            # skip files that don't match pattern (or optionally group them by filename prefix)
            print(f"⚠️ Skipping file with unexpected name format: {f}")
            continue
        subj = m.group(1).zfill(3)  # pad subject id to 3 digits for consistent labels
        eye = m.group(2).upper()
        key = f"{subj}_{eye}"
        groups.setdefault(key, []).append(f)

    # helper to convert a single file into a 1D float32 normalized vector
    def file_to_vector(path):
        full = os.path.join(root_folder, path)
        if full.lower().endswith('.npz'):
            data = np.load(full, allow_pickle=True)
            if 'iris_code' not in data.files or 'mask_code' not in data.files:
                raise ValueError(f"{path} missing 'iris_code' or 'mask_code' keys.")
            iris_code = np.array(data['iris_code']).reshape(-1)
            mask_code = np.array(data['mask_code']).reshape(-1)
            if iris_code.shape != mask_code.shape:
                # try broadcasting or raise
                if iris_code.size == mask_code.size:
                    pass
                else:
                    raise ValueError(f"Shape mismatch in {path}: iris_code {iris_code.shape}, mask_code {mask_code.shape}")
            # mask==1 => occluded/unreliable; set corresponding bits to 0
            vec = np.where(mask_code == 1, 0, iris_code).astype(np.float32)
        elif full.lower().endswith('.npy'):
            vec = np.load(full).reshape(-1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported file: {path}")

        # L2 normalize if possible
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    index_vectors = []
    index_labels = []
    query_vectors = []
    query_labels = []

    # iterate groups
    for key in sorted(groups.keys()):
        group_files = groups[key][:]
        if len(group_files) == 0:
            continue

        if pick == 'random':
            random.shuffle(group_files)
        elif pick == 'first':
            group_files = sorted(group_files)
        else:
            raise ValueError("pick must be 'random' or 'first'")

        # pick first as index, rest are query
        idx_file = group_files[0]
        try:
            idx_vec = file_to_vector(idx_file)
        except Exception as e:
            print(f"⚠️ Failed to load index file {idx_file} for {key}: {e}. Skipping this group.")
            continue

        index_vectors.append(idx_vec)
        index_labels.append(key)

        for qf in group_files[1:]:
            try:
                qvec = file_to_vector(qf)
            except Exception as e:
                print(f"⚠️ Failed to load query file {qf} for {key}: {e}. Skipping this file.")
                continue
            query_vectors.append(qvec)
            query_labels.append(key)

        print(f"{key}: index=1, query={max(0, len(group_files)-1)}")

    if len(index_vectors) == 0:
        raise ValueError("No index vectors found. Check filename patterns and folder contents.")

    index_vectors = np.vstack(index_vectors).astype(np.float32)
    index_labels = np.array(index_labels)

    if len(query_vectors) > 0:
        query_vectors = np.vstack(query_vectors).astype(np.float32)
        query_labels = np.array(query_labels)
    else:
        D = index_vectors.shape[1]
        query_vectors = np.zeros((0, D), dtype=np.float32)
        query_labels = np.array([])

    return index_vectors, index_labels, query_vectors, query_labels


    


# %%
data_folder = "/home/nishkal/sg/iris_indexing/CASIA-Iris-Lamp_/outputs_npz/templates"
X_index, y_index, X_query, y_query = load_IrisLamp_templates(data_folder, seed=42)

hnsw = HNSWPL(M=16, ef_construction=200, random_seed=42)

for i, vec in enumerate(X_index):
        hnsw.add_item(vec, idx=i)

# %%
results = evaluate_top1_timing(hnsw, y_index, X_query, y_query,ef_search=150)
print("Hit rate:", results["hit_rate"])
print("Average search time (ms):", results["avg_time_s"] * 1000)

# %%
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_density_scores(X, k=10):
    """
    Compute density score for each vector.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset vectors (N x D)
    k : int
        Number of neighbors
    
    Returns
    -------
    density_scores : np.ndarray
    """
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(X)
    
    distances, _ = nbrs.kneighbors(X)
    
    # remove self-distance
    distances = distances[:, 1:]
    
    density_scores = np.mean(distances, axis=1)
    
    return density_scores


def normalize_density(density_scores):
    
    min_d = density_scores.min()
    max_d = density_scores.max()
    
    norm = (density_scores - min_d) / (max_d - min_d + 1e-9)
    
    return norm




# %%
class Node:
    def __init__(self, idx: int, vector: np.ndarray, level: int):
        self.idx = idx
        self.vector = vector
        self.level = level
        self.neighbors = [[] for _ in range(level + 1)]
        self.deleted = False


class HNSWDA:
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

    def get_random_level(self, density_score):
        
        r = random.random()
        
        if r == 0.0:
            r = 1e-16
        
        base_level = -math.log(r) * self.level_mult
        
        adjusted_level = base_level * (1 + density_score)
        
        return int(adjusted_level)


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

    def add_item(self, vector, density_score, idx=None):
        if idx is None:
            idx = len(self.nodes)

        vector = np.asarray(vector, dtype=float)

        level = self.get_random_level(density_score)

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


# %%
data_folder = "/home/nishkal/sg/iris_indexing/CASIA-Iris-Lamp_/outputs_npz/templates"
X_index, y_index, X_query, y_query = load_IrisLamp_templates(data_folder, seed=42)

density_scores = compute_density_scores(X_index, k=10)

density_scores = normalize_density(density_scores)

hnswda = HNSWDA(M=16, ef_construction=200, random_seed=42)

for i, vec in enumerate(X_index):
    hnswda.add_item(vec, density_scores[i], idx=i)

# %%
results = evaluate_top1_timing(hnswda, y_index, X_query, y_query,ef_search=150)
print("Hit rate:", results["hit_rate"])
print("Average search time (ms):", results["avg_time_s"] * 1000)

# %%
import os
import random
import numpy as np


def load_iris_txt_templates(root_folder, seed=42, pick='random', normalize=True):
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
    - pick 1 pair for indexing
    - use remaining pairs for querying

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
        """
        Reads text file containing binary bits.
        Supports:
        - 0 1 1 0 1
        - 0110101
        - multi-line binary rows
        Returns flattened uint8 array.
        """
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

        # Since mask=1 means valid, mask=0 means invalid:
        vec = (template & mask).astype(np.float32)

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec

    index_vectors = []
    index_labels = []
    query_vectors = []
    query_labels = []

    for subject in subject_folders:
        subject_path = os.path.join(root_folder, subject)

        pairs = []
        for i in range(1, 11):
            template_file = os.path.join(subject_path, f"{i}_template.txt")
            mask_file = os.path.join(subject_path, f"{i}_mask.txt")

            if os.path.exists(template_file) and os.path.exists(mask_file):
                pairs.append((i, template_file, mask_file))

        if len(pairs) == 0:
            print(f"⚠️ No valid txt pairs found for subject {subject}, skipping.")
            continue

        if pick == 'random':
            random.shuffle(pairs)
        elif pick == 'first':
            pairs = sorted(pairs, key=lambda x: x[0])
        else:
            raise ValueError("pick must be 'random' or 'first'")

        idx_num, idx_template, idx_mask = pairs[0]

        try:
            idx_vec = pair_to_vector(idx_template, idx_mask)
            index_vectors.append(idx_vec)
            index_labels.append(subject)
        except Exception as e:
            print(f"⚠️ Failed index pair for subject {subject}, image {idx_num}: {e}")
            continue

        q_count = 0
        for q_num, q_template, q_mask in pairs[1:]:
            try:
                q_vec = pair_to_vector(q_template, q_mask)
                query_vectors.append(q_vec)
                query_labels.append(subject)
                q_count += 1
            except Exception as e:
                print(f"⚠️ Failed query pair for subject {subject}, image {q_num}: {e}")

        print(f"{subject}: index=1, query={q_count}")

    if len(index_vectors) == 0:
        raise ValueError("No index vectors created. Check txt file contents.")

    index_vectors = np.vstack(index_vectors).astype(np.float32)
    index_labels = np.array(index_labels)

    if len(query_vectors) > 0:
        query_vectors = np.vstack(query_vectors).astype(np.float32)
        query_labels = np.array(query_labels)
    else:
        D = index_vectors.shape[1]
        query_vectors = np.zeros((0, D), dtype=np.float32)
        query_labels = np.array([])

    return index_vectors, index_labels, query_vectors, query_labels

# %%
import os
import numpy as np
from PIL import Image


def read_txt_bits(path):
    """
    Read binary bits from a txt file.
    Supports formats like:
    0 1 1 0 1
    0110101
    multi-line rows of 0/1
    """
    with open(path, "r") as f:
        content = f.read()

    bits = [int(ch) for ch in content if ch in "01"]

    if len(bits) == 0:
        raise ValueError(f"No binary digits found in {path}")

    return np.array(bits, dtype=np.uint8)


def save_feat_txt(feat_vector, out_path, row_length=None):
    """
    Save feature vector to txt.
    If row_length is given, reshape for nicer formatting.
    """
    feat_vector = feat_vector.astype(np.uint8)

    with open(out_path, "w") as f:
        if row_length is None:
            f.write("".join(map(str, feat_vector.tolist())))
        else:
            for i in range(0, len(feat_vector), row_length):
                row = feat_vector[i:i + row_length]
                f.write("".join(map(str, row.tolist())) + "\n")


def save_feat_bmp(feat_vector, out_path, shape=None):
    """
    Save feature vector as BMP image.
    0 -> black, 1 -> white
    """
    feat_vector = feat_vector.astype(np.uint8)

    if shape is None:
        # default: make it a single-row image
        arr = feat_vector.reshape(1, -1)
    else:
        if np.prod(shape) != feat_vector.size:
            raise ValueError(
                f"Cannot reshape feature of size {feat_vector.size} into shape {shape}"
            )
        arr = feat_vector.reshape(shape)

    img = Image.fromarray(arr * 255)  # binary to grayscale image
    img.save(out_path)


def create_combined_features(root_folder, save_txt=True, save_bmp=True):
    """
    For each subject folder in root_folder:
    - reads i_template.txt and i_mask.txt
    - computes feat = template & mask
    - saves:
        i_feat.txt
        i_feat.bmp

    Assumption:
    - mask == 1 means valid
    - mask == 0 means invalid
    - final feature = template & mask
    """

    if not os.path.isdir(root_folder):
        raise ValueError(f"Root folder not found: {root_folder}")

    subject_folders = sorted([
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ])

    if not subject_folders:
        raise ValueError(f"No subject folders found in {root_folder}")

    total_done = 0
    total_failed = 0

    for subject in subject_folders:
        subject_path = os.path.join(root_folder, subject)
        print(f"\nProcessing subject {subject}...")

        for i in range(1, 11):
            template_txt = os.path.join(subject_path, f"{i}_template.txt")
            mask_txt = os.path.join(subject_path, f"{i}_mask.txt")

            feat_txt = os.path.join(subject_path, f"{i}_feat.txt")
            feat_bmp = os.path.join(subject_path, f"{i}_feat.bmp")

            if not (os.path.exists(template_txt) and os.path.exists(mask_txt)):
                print(f"  ⚠️ Missing pair for image {i}, skipping.")
                total_failed += 1
                continue

            try:
                template = read_txt_bits(template_txt)
                mask = read_txt_bits(mask_txt)

                if template.shape != mask.shape:
                    raise ValueError(
                        f"Shape mismatch: template {template.shape}, mask {mask.shape}"
                    )

                feat = template & mask   # mask=1 valid, mask=0 invalid

                if save_txt:
                    # Save as one long row; you can change row_length if needed
                    save_feat_txt(feat, feat_txt, row_length=None)

                if save_bmp:
                    # Try to use original bmp shape if available
                    template_bmp = os.path.join(subject_path, f"{i}_template.bmp")
                    if os.path.exists(template_bmp):
                        img = Image.open(template_bmp).convert("L")
                        shape = np.array(img).shape
                    else:
                        shape = (1, feat.size)

                    save_feat_bmp(feat, feat_bmp, shape=shape)

                print(f"  ✅ Image {i}: saved feat.txt and feat.bmp")
                total_done += 1

            except Exception as e:
                print(f"  ❌ Image {i} failed: {e}")
                total_failed += 1

    print("\nDone.")
    print(f"Total processed successfully: {total_done}")
    print(f"Total failed/skipped: {total_failed}")

# %%
root_folder = "/home/nishkal/sg/iris_indexing/datasets/iris_syn"
create_combined_features(root_folder)

# %%
root_folder = "/home/nishkal/sg/iris_indexing/datasets/iris_syn"

index_vectors, index_labels, query_vectors, query_labels = load_iris_txt_templates(
    root_folder=root_folder,
    seed=42,
    pick='random'
)

# print(index_vectors.shape)
# print(index_labels.shape)
# print(query_vectors.shape)
# print(query_labels.shape)

# %%
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


# %%
hnsw = HNSW(M=32, ef_construction=300, random_seed=42)

for i, vec in enumerate(index_vectors):
        hnsw.add_item(vec, idx=i)

# %%
results = evaluate_top1_timing(hnsw,index_labels,query_vectors,query_labels,ef_search=400)
print("Hit rate:", results["hit_rate"])
print("Average search time (ms):", results["avg_time_s"] * 1000)

# %%
import numpy as np
import hnswlib

def evaluate_hnsw_accuracy(
    index_vectors,
    index_labels,
    query_vectors,
    query_labels,
    k=1,
    space='l2',
    M=16,
    ef_construction=200,
    ef_search=100
):
    """
    Build HNSW index and evaluate retrieval accuracy.

    Parameters
    ----------
    index_vectors : np.ndarray, shape (N, d)
    index_labels : np.ndarray, shape (N,)
        Identity label for each index vector
    query_vectors : np.ndarray, shape (Q, d)
    query_labels : np.ndarray, shape (Q,)
        Ground-truth identity label for each query
    k : int
        Number of nearest neighbors to retrieve
    space : str
        Distance space: 'l2', 'ip', or 'cosine'
    M : int
        HNSW graph connectivity
    ef_construction : int
        Construction-time search width
    ef_search : int
        Query-time search width

    Returns
    -------
    accuracy : float
        Rank-1 accuracy if k=1, otherwise top-k accuracy
    retrieved_labels : np.ndarray
        Labels of retrieved neighbors, shape (Q, k)
    neighbor_ids : np.ndarray
        Internal returned ids from HNSW, shape (Q, k)
    distances : np.ndarray
        Distances to retrieved neighbors, shape (Q, k)
    """
    index_vectors = np.asarray(index_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    index_labels = np.asarray(index_labels)
    query_labels = np.asarray(query_labels)

    num_elements, dim = index_vectors.shape

    # Create HNSW index
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(
        max_elements=num_elements,
        ef_construction=ef_construction,
        M=M
    )

    # Add vectors with ids = 0..N-1
    ids = np.arange(num_elements)
    index.add_items(index_vectors, ids)

    # Set search parameter
    index.set_ef(ef_search)

    # Search
    neighbor_ids, distances = index.knn_query(query_vectors, k=k)

    # Convert returned ids to actual person labels
    retrieved_labels = index_labels[neighbor_ids]

    # Rank-1 accuracy
    if k == 1:
        pred_labels = retrieved_labels[:, 0]
        accuracy = np.mean(pred_labels == query_labels)
    else:
        # Top-k accuracy: true label appears anywhere in top-k
        correct = np.any(retrieved_labels == query_labels[:, None], axis=1)
        accuracy = np.mean(correct)

    return accuracy, retrieved_labels, neighbor_ids, distances

# %%
import numpy as np
from sklearn.neighbors import NearestNeighbors

def evaluate_nn_accuracy(
    index_vectors,
    index_labels,
    query_vectors,
    query_labels,
    k=1,
    metric='euclidean'
):
    index_vectors = np.asarray(index_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    index_labels = np.asarray(index_labels)
    query_labels = np.asarray(query_labels)

    nn = NearestNeighbors(
        n_neighbors=k,
        metric=metric,
        algorithm='auto'
    )
    nn.fit(index_vectors)

    distances, neighbor_ids = nn.kneighbors(query_vectors)
    retrieved_labels = index_labels[neighbor_ids]

    if k == 1:
        pred_labels = retrieved_labels[:, 0]
        accuracy = np.mean(pred_labels == query_labels)
    else:
        accuracy = np.mean(
            np.any(retrieved_labels == query_labels[:, None], axis=1)
        )

    return accuracy, retrieved_labels, neighbor_ids, distances

# %%
acc1, retrieved_labels_1, neighbor_ids_1, distances_1 = evaluate_nn_accuracy(
    index_vectors=index_vectors,
    index_labels=index_labels,
    query_vectors=query_vectors,
    query_labels=query_labels,
    k=1,
    metric='euclidean'   # or 'cosine'
)

print(f"Rank-1 Accuracy: {acc1 * 100:.2f}%")


