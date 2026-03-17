# exploring faiss algos
import os
import numpy as np
import faiss
from scipy.io import loadmat

# Path to your .mat files
data_folder = "templates/CASIA1/features"

# Get all .mat files for left & right eye templates (_1_1, _2_1)
files = sorted([f for f in os.listdir(data_folder) if f.endswith('.mat')])

vectors = []       # store eye vectors
labels = []        # store labels (person ID + eye side)

for file in files:
    # Only take enrollment templates (1 or 2, not 3)
    if '_3' in file:
        continue
    
    path = os.path.join(data_folder, file)
    data = loadmat(path)
    
    # Replace 'feature_vector' with your actual key name
    vec = data['template'].flatten().astype('float32')
    
    # Extract person ID and eye side
    pid, side = file.split('_')[0], file.split('_')[1]  # e.g., '000', '1' or '2'
    labels.append(f"{pid}_{side}")  # e.g., "000_1" for left eye
    
    vectors.append(vec)

# Convert to numpy array (n, d)
vectors = np.vstack(vectors)

# ---- Create HNSW Index ----
d = vectors.shape[1]  # dimension of a single eye vector
M = 32
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 100
index.add(vectors)
index.hnsw.efSearch = 50

# ---- Accuracy Evaluation ----
query_files = [f for f in os.listdir(data_folder) if '_3' in f and f.endswith('.mat')]

correct = 0
total = 0
k = 54  # Top-k

for qf in sorted(query_files):
    qpid = qf.split('_')[0]  # true person ID
    qpath = os.path.join(data_folder, qf)
    
    qvec = loadmat(qpath)['template'].flatten().astype('float32').reshape(1, -1)
    
    distances, indices = index.search(qvec, k)  # retrieve top-10
    predicted_pids = [labels[i].split('_')[0] for i in indices[0]]  # extract IDs
    
    if qpid in predicted_pids:
        correct += 1
    total += 1

accuracy_top10 = correct / total if total > 0 else 0
penetration_rate = k/total
print(f"penetration rate: {penetration_rate}")
print(correct,total)
print(f"Top-10 Accuracy: {accuracy_top10*100:.2f}%")