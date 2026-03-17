import os
import numpy as np
import scipy.io
import faiss
import plotly.graph_objects as go

def preprocess_dataset(mat_dir, feature_key='template'):
    """Load features from .mat files and organize them for each person."""
    data = {}
    for file in os.listdir(mat_dir):
        if file.endswith('.mat'):
            person_id = file.split('_')[0]
            mat = scipy.io.loadmat(os.path.join(mat_dir, file))
            feature = mat.get(feature_key)
            if feature is not None:
                feature = feature.flatten().astype('float32')
                data.setdefault(person_id, []).append((file, feature))
    return data

def build_hnsw_index(data, metric='l2', ef_construction=200, M=32):
    """Build FAISS HNSW index from dataset using features of type 1 and 2."""
    person_ids = []
    vectors = []

    for pid, files in data.items():
        # Only take _1 and _2 as database features
        db_feats = [f for name, f in files if '_1' in name or '_2' in name]
        if db_feats:
            vectors.append(np.mean(db_feats, axis=0))  # averaging both eyes (no concatenation)
            person_ids.append(pid)

    vectors = np.vstack(vectors).astype('float32')

    d = vectors.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)

    return index, person_ids

def evaluate_hit_rate_vs_k(index, data, person_ids, max_k=None):
    """Evaluate hit rate for each k from 1 to max_k (default = total people)."""
    query_features = []
    query_ids = []

    for pid, files in data.items():
        # Only take _3 as query
        q_feats = [f for name, f in files if '_3' in name]
        if q_feats:
            query_features.append(np.mean(q_feats, axis=0))
            query_ids.append(pid)

    query_features = np.vstack(query_features).astype('float32')

    total_queries = len(query_ids)
    max_k = max_k or len(person_ids)

    hit_rates = []
    for k in range(1, max_k + 1):
        D, I = index.search(query_features, k)
        correct = sum(query_ids[i] in [person_ids[j] for j in row] for i, row in enumerate(I))
        hit_rate = correct / total_queries
        hit_rates.append(hit_rate)

    accuracy_k1 = hit_rates[0]  # Accuracy is just the hit rate at k=1
    return hit_rates, accuracy_k1

def plot_hit_rate(hit_rates):
    """Plot hit rate vs k using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(hit_rates) + 1)),
        y=hit_rates,
        mode='lines+markers',
        name='Hit Rate'
    ))
    fig.update_layout(
        title="Hit Rate vs K",
        xaxis_title="K (Top-K)",
        yaxis_title="Hit Rate",
        template="plotly_white"
    )
    fig.show()

# Usage Example
mat_dir = "templates/CASIA1/features"
data = preprocess_dataset(mat_dir, feature_key='DeepFeature')
index, person_ids = build_hnsw_index(data)
hit_rates, accuracy_k1 = evaluate_hit_rate_vs_k(index, data, person_ids)
# plot_hit_rate(hit_rates)

print(f"Top-1 Accuracy: {accuracy_k1:.4f}")
