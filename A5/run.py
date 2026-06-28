import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from sklearn.manifold import TSNE
from itertools import combinations
from tqdm import tqdm
import random, os, time, logging, argparse
import urllib.request, zipfile

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GENRE_COLS = ['unknown','Action','Adventure','Animation',"Children's",
              'Comedy','Crime','Documentary','Drama','Fantasy',
              'Film-Noir','Horror','Musical','Mystery','Romance',
              'Sci-Fi','Thriller','War','Western']

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

#Logging
def get_logger(name):
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(f'logs/{name}.log')
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger

#Argument parser
def parse_args():
    p = argparse.ArgumentParser(description='A5: Graph Neural Networks')
    p.add_argument('--model', choices=['gcn','gat','sage','mlp','all'], default='all',
                   help='Which model to train')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--heads', type=int, default=8, help='GAT attention heads')
    p.add_argument('--k', type=int, default=10, help='GraphSAGE neighbor sample size')
    p.add_argument('--ex1', action='store_true', help='Exercise 1: over-smoothing')
    p.add_argument('--ex2', action='store_true', help='Exercise 2: model comparison + attention')
    p.add_argument('--ex3', action='store_true', help='Exercise 3: MLP baseline')
    p.add_argument('--all-ex', action='store_true', help='Run all exercises')
    p.add_argument('--rec', action='store_true', help='Run recommendation system')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def load_movielens(logger, min_common=5):
    os.makedirs('data/movielens', exist_ok=True)
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    if not os.path.exists('data/movielens/ml-100k/u.data'):
        logger.info('Downloading MovieLens-100k...')
        urllib.request.urlretrieve(url, 'data/movielens/ml-100k.zip')
        with zipfile.ZipFile('data/movielens/ml-100k.zip') as z:
            z.extractall('data/movielens/')
        logger.info('Download complete.')

    ratings = pd.read_csv('data/movielens/ml-100k/u.data',
                          sep='\t', names=['user','item','rating','timestamp'])
    movies  = pd.read_csv('data/movielens/ml-100k/u.item', sep='|',
                          encoding='latin-1', header=None,
                          names=['item','title','release_date','video_date',
                                 'imdb_url'] + GENRE_COLS)

    # Node features: 18 genre flags + normalized release year
    movies['year'] = movies['release_date'].str.extract(r'(\d{4})').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    movies['year_norm'] = ((movies['year'] - movies['year'].min()) /
                           (movies['year'].max() - movies['year'].min()))

    feat_cols      = GENRE_COLS + ['year_norm']
    movie_features = movies[feat_cols].fillna(0).values.astype(np.float32)

    movie_ids = sorted(movies.item.unique())
    mid2idx   = {m: i for i, m in enumerate(movie_ids)}
    N_MOVIES  = len(movie_ids)

    # Labels: primary genre = first genre column that is 1
    movies_indexed = movies.set_index('item').loc[movie_ids]
    genre_matrix   = movies_indexed[GENRE_COLS].values
    labels_raw = []
    for row in genre_matrix:
        idx = int(np.argmax(row))
        if row[idx] == 0:
            idx = 0
        labels_raw.append(idx)
    labels    = np.array(labels_raw)
    n_classes = len(set(labels))

    logger.info(f'Movies: {N_MOVIES} | Feature dim: {movie_features.shape[1]} | Classes: {n_classes}')

    # Build co-rating graph
    user_movies = ratings.groupby('user')['item'].apply(list)
    rows, cols  = [], []
    for items in user_movies:
        valid = [mid2idx[m] for m in items if m in mid2idx]
        for a, b in combinations(valid, 2):
            rows.append(a); cols.append(b)
            rows.append(b); cols.append(a)

    edge_df      = pd.DataFrame({'row': rows, 'col': cols})
    edge_counts  = edge_df.groupby(['row','col']).size().reset_index(name='count')
    strong_edges = edge_counts[edge_counts['count'] >= min_common]
    logger.info(f'Co-rating edges (>= {min_common} users): {len(strong_edges):,}')

    A_data = np.ones(len(strong_edges))
    A_coo  = sp.coo_matrix((A_data,
                             (strong_edges['row'].values, strong_edges['col'].values)),
                            shape=(N_MOVIES, N_MOVIES))
    A_movie = torch.FloatTensor(A_coo.toarray()).to(device)
    X_movie = torch.FloatTensor(movie_features).to(device)
    Y_movie = torch.LongTensor(labels).to(device)

    # Adjacency list for GraphSAGE
    A_np     = A_coo.toarray()
    adj_list = [list(np.where(A_np[i] > 0)[0]) for i in range(N_MOVIES)]

    # Train/val/test split: up to 20 per class for train
    train_mask = torch.zeros(N_MOVIES, dtype=torch.bool)
    val_mask   = torch.zeros(N_MOVIES, dtype=torch.bool)
    test_mask  = torch.zeros(N_MOVIES, dtype=torch.bool)

    for c in range(n_classes):
        idx = (Y_movie == c).nonzero(as_tuple=True)[0]
        if len(idx) >= 20:
            train_mask[idx[:20]] = True
        elif len(idx) > 0:
            train_mask[idx[:len(idx)//2]] = True

    remaining = (~train_mask).nonzero(as_tuple=True)[0]
    n_val = min(200, len(remaining)//2)
    val_mask[remaining[:n_val]]           = True
    test_mask[remaining[n_val:n_val+500]] = True

    logger.info(f'Split - Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}')

    return (X_movie, Y_movie, A_movie, A_coo, adj_list,
            n_classes, N_MOVIES, movie_ids, mid2idx,
            train_mask.to(device), val_mask.to(device), test_mask.to(device),
            ratings, movies)

#Data Loading
def load_movielens(logger, min_common=5):
    os.makedirs('data/movielens', exist_ok=True)
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    if not os.path.exists('data/movielens/ml-100k/u.data'):
        logger.info('Downloading MovieLens-100k...')
        urllib.request.urlretrieve(url, 'data/movielens/ml-100k.zip')
        with zipfile.ZipFile('data/movielens/ml-100k.zip') as z:
            z.extractall('data/movielens/')
        logger.info('Download complete.')

    ratings = pd.read_csv('data/movielens/ml-100k/u.data',
                          sep='\t', names=['user','item','rating','timestamp'])
    movies  = pd.read_csv('data/movielens/ml-100k/u.item', sep='|',
                          encoding='latin-1', header=None,
                          names=['item','title','release_date','video_date',
                                 'imdb_url'] + GENRE_COLS)

    movies['year'] = movies['release_date'].str.extract(r'(\d{4})').astype(float)
    movies['year'] = movies['year'].fillna(movies['year'].median())
    movies['year_norm'] = ((movies['year'] - movies['year'].min()) /
                           (movies['year'].max() - movies['year'].min()))

    feat_cols      = GENRE_COLS + ['year_norm']
    movie_features = movies[feat_cols].fillna(0).values.astype(np.float32)

    movie_ids = sorted(movies.item.unique())
    mid2idx   = {m: i for i, m in enumerate(movie_ids)}
    N_MOVIES  = len(movie_ids)

    movies_indexed = movies.set_index('item').loc[movie_ids]
    genre_matrix   = movies_indexed[GENRE_COLS].values
    labels_raw = []
    for row in genre_matrix:
        idx = int(np.argmax(row))
        if row[idx] == 0:
            idx = 0
        labels_raw.append(idx)
    labels    = np.array(labels_raw)
    n_classes = len(set(labels))

    logger.info(f'Movies: {N_MOVIES} | Feature dim: {movie_features.shape[1]} | Classes: {n_classes}')

    user_movies = ratings.groupby('user')['item'].apply(list)
    rows, cols  = [], []
    for items in user_movies:
        valid = [mid2idx[m] for m in items if m in mid2idx]
        for a, b in combinations(valid, 2):
            rows.append(a); cols.append(b)
            rows.append(b); cols.append(a)

    edge_df      = pd.DataFrame({'row': rows, 'col': cols})
    edge_counts  = edge_df.groupby(['row','col']).size().reset_index(name='count')
    strong_edges = edge_counts[edge_counts['count'] >= min_common]
    logger.info(f'Co-rating edges (>= {min_common} users): {len(strong_edges):,}')

    A_data = np.ones(len(strong_edges))
    A_coo  = sp.coo_matrix((A_data,
                             (strong_edges['row'].values, strong_edges['col'].values)),
                            shape=(N_MOVIES, N_MOVIES))
    A_movie = torch.FloatTensor(A_coo.toarray()).to(device)
    X_movie = torch.FloatTensor(movie_features).to(device)
    Y_movie = torch.LongTensor(labels).to(device)

    A_np     = A_coo.toarray()
    adj_list = [list(np.where(A_np[i] > 0)[0]) for i in range(N_MOVIES)]

    train_mask = torch.zeros(N_MOVIES, dtype=torch.bool)
    val_mask   = torch.zeros(N_MOVIES, dtype=torch.bool)
    test_mask  = torch.zeros(N_MOVIES, dtype=torch.bool)

    for c in range(n_classes):
        idx = (Y_movie == c).nonzero(as_tuple=True)[0]
        if len(idx) >= 20:
            train_mask[idx[:20]] = True
        elif len(idx) > 0:
            train_mask[idx[:len(idx)//2]] = True

    remaining = (~train_mask).nonzero(as_tuple=True)[0]
    n_val = min(200, len(remaining)//2)
    val_mask[remaining[:n_val]]           = True
    test_mask[remaining[n_val:n_val+500]] = True

    logger.info(f'Split - Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}')

    return (X_movie, Y_movie, A_movie, A_coo, adj_list,
            n_classes, N_MOVIES, movie_ids, mid2idx,
            train_mask.to(device), val_mask.to(device), test_mask.to(device),
            ratings, movies)

#Graph Utilities
def normalize_adjacency(A):
    A_tilde = A + torch.eye(A.size(0), device=A.device)
    D = A_tilde.sum(dim=1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

#GCN
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H, A_norm):
        return A_norm @ self.W(H)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        dims = [in_features] + [hidden_dim] * (n_layers - 1) + [n_classes]
        self.layers  = nn.ModuleList([GCNLayer(dims[i], dims[i+1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_norm):
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(h, A_norm)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
        emb = h
        return h, emb

#GAT
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W           = nn.Linear(in_features, out_features, bias=False)
        self.a           = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu  = nn.LeakyReLU(alpha)
        self.dropout     = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, H, A):
        N  = H.size(0)
        Wh = self.W(H)
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)
        e    = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        mask = (A == 0) & (~torch.eye(N, dtype=torch.bool, device=A.device))
        e    = e.masked_fill(mask, float('-inf'))
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        out  = alpha @ Wh
        return out, alpha

class GAT(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, n_heads=8, dropout=0.6):
        super().__init__()
        self.heads     = nn.ModuleList([
            GATLayer(in_features, hidden_dim, dropout) for _ in range(n_heads)
        ])
        self.out_layer = GATLayer(hidden_dim * n_heads, n_classes, dropout)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, X, A):
        X    = self.dropout(X)
        outs = [F.elu(head(X, A)[0]) for head in self.heads]
        h    = torch.cat(outs, dim=-1)
        h    = self.dropout(h)
        out, attn = self.out_layer(h, A)
        return out, h, attn

#GraphSAGE
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_features)

    def forward(self, H, adj_list, k=10):
        N   = H.size(0)
        agg = torch.zeros(N, H.size(1), device=H.device)
        for v in range(N):
            nbrs = adj_list[v]
            if len(nbrs) == 0:
                agg[v] = H[v]
            else:
                sampled = random.choices(nbrs, k=min(k, len(nbrs)))
                agg[v]  = H[torch.tensor(sampled, device=H.device)].mean(dim=0)
        return F.relu(self.W(torch.cat([H, agg], dim=-1)))

class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, k=10, dropout=0.5):
        super().__init__()
        self.layer1  = GraphSAGELayer(in_features, hidden_dim)
        self.layer2  = GraphSAGELayer(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.k       = k

    def forward(self, X, adj_list):
        h   = self.layer1(X, adj_list, self.k)
        h   = self.dropout(h)
        out = self.layer2(h, adj_list, self.k)
        return out, h

#MLP (no graph)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, n_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, X):
        return self.net(X)

#Training Functions
def train_gcn(X, Y, A_norm, train_mask, val_mask, test_mask,
              n_classes, args, logger, n_layers=2, tag='gcn'):
    model = GCN(X.shape[1], args.hidden, n_classes,
                n_layers=n_layers, dropout=args.dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    train_accs, val_accs, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        logits, _ = model(X, A_norm)
        loss = F.cross_entropy(logits[train_mask], Y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(X, A_norm)
            tr_acc = (logits[train_mask].argmax(1) == Y[train_mask]).float().mean().item()
            va_acc = (logits[val_mask].argmax(1)   == Y[val_mask]).float().mean().item()
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        epoch_times.append(time.time() - t0)
        if (epoch + 1) % 50 == 0:
            logger.info(f'[{tag}] Epoch {epoch+1:3d} | Loss: {loss:.4f} | '
                        f'Train: {tr_acc:.4f} | Val: {va_acc:.4f} | '
                        f'Time: {epoch_times[-1]*1000:.0f}ms')

    model.eval()
    with torch.no_grad():
        logits, emb = model(X, A_norm)
        test_acc = (logits[test_mask].argmax(1) == Y[test_mask]).float().mean().item()
    logger.info(f'[{tag}] Test Accuracy: {test_acc*100:.2f}%')
    return model, emb, test_acc, train_accs, val_accs, epoch_times


def train_gat(X, Y, A_movie, train_mask, val_mask, test_mask,
              n_classes, args, logger):
    model = GAT(X.shape[1], args.hidden // args.heads, n_classes,
                n_heads=args.heads, dropout=0.6).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    val_accs, epoch_times = [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        logits, _, _ = model(X, A_movie)
        loss = F.cross_entropy(logits[train_mask], Y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X, A_movie)
            va_acc = (logits[val_mask].argmax(1) == Y[val_mask]).float().mean().item()
        val_accs.append(va_acc)
        epoch_times.append(time.time() - t0)
        if (epoch + 1) % 50 == 0:
            logger.info(f'[gat] Epoch {epoch+1:3d} | Loss: {loss:.4f} | '
                        f'Val: {va_acc:.4f} | Time: {epoch_times[-1]*1000:.0f}ms')

    model.eval()
    with torch.no_grad():
        logits, emb, attn = model(X, A_movie)
        test_acc = (logits[test_mask].argmax(1) == Y[test_mask]).float().mean().item()
    logger.info(f'[gat] Test Accuracy: {test_acc*100:.2f}%')
    return model, emb, attn, test_acc, val_accs, epoch_times


def train_sage(X, Y, adj_list, train_mask, val_mask, test_mask,
               n_classes, args, logger):
    model = GraphSAGE(X.shape[1], args.hidden, n_classes,
                      k=args.k, dropout=args.dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    train_accs, val_accs, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        logits, _ = model(X, adj_list)
        loss = F.cross_entropy(logits[train_mask], Y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(X, adj_list)
            tr_acc = (logits[train_mask].argmax(1) == Y[train_mask]).float().mean().item()
            va_acc = (logits[val_mask].argmax(1)   == Y[val_mask]).float().mean().item()
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        epoch_times.append(time.time() - t0)
        if (epoch + 1) % 50 == 0:
            logger.info(f'[sage] Epoch {epoch+1:3d} | Loss: {loss:.4f} | '
                        f'Train: {tr_acc:.4f} | Val: {va_acc:.4f} | '
                        f'Time: {epoch_times[-1]*1000:.0f}ms')

    model.eval()
    with torch.no_grad():
        logits, emb = model(X, adj_list)
        test_acc = (logits[test_mask].argmax(1) == Y[test_mask]).float().mean().item()
    logger.info(f'[sage] Test Accuracy: {test_acc*100:.2f}%')
    return model, emb, test_acc, train_accs, val_accs, epoch_times


def train_mlp(X, Y, train_mask, val_mask, test_mask,
              n_classes, args, logger):
    model = MLP(X.shape[1], args.hidden, n_classes, args.dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    val_accs, epoch_times = [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        logits = model(X)
        loss   = F.cross_entropy(logits[train_mask], Y[train_mask])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X)
            va_acc = (logits[val_mask].argmax(1) == Y[val_mask]).float().mean().item()
        val_accs.append(va_acc)
        epoch_times.append(time.time() - t0)
        if (epoch + 1) % 50 == 0:
            logger.info(f'[mlp] Epoch {epoch+1:3d} | Loss: {loss:.4f} | '
                        f'Val: {va_acc:.4f} | Time: {epoch_times[-1]*1000:.0f}ms')

    model.eval()
    with torch.no_grad():
        logits = model(X)
        test_acc = (logits[test_mask].argmax(1) == Y[test_mask]).float().mean().item()
    logger.info(f'[mlp] Test Accuracy: {test_acc*100:.2f}%')
    return model, test_acc, val_accs, epoch_times

#Exercise 1: Over-smoothing
def exercise1(X, Y, A_norm, train_mask, val_mask, test_mask, n_classes, args, logger):
    logger.info('=== Exercise 1: Over-smoothing Analysis ===')
    results = []
    for n_layers in [1, 2, 3, 4, 5]:
        _, emb, test_acc, _, _, _ = train_gcn(
            X, Y, A_norm, train_mask, val_mask, test_mask,
            n_classes, args, logger, n_layers=n_layers, tag=f'gcn-L{n_layers}')

        emb_np = emb[test_mask].cpu().detach().numpy()
        # Pairwise cosine similarity on test embeddings
        norms  = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8
        emb_n  = emb_np / norms
        sim_matrix = emb_n @ emb_n.T
        # Average off-diagonal similarity
        n = sim_matrix.shape[0]
        mask_diag = ~np.eye(n, dtype=bool)
        avg_sim = sim_matrix[mask_diag].mean()

        results.append((n_layers, test_acc, avg_sim))
        logger.info(f'  Layers={n_layers} | Test Acc: {test_acc*100:.2f}% | Avg Cosine Sim: {avg_sim:.4f}')

    # Plot
    layers_list = [r[0] for r in results]
    accs        = [r[1]*100 for r in results]
    sims        = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = 'steelblue'
    ax1.set_xlabel('Number of GCN Layers')
    ax1.set_ylabel('Test Accuracy (%)', color=color1)
    ax1.plot(layers_list, accs, 'o-', color=color1, label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tomato'
    ax2.set_ylabel('Avg Pairwise Cosine Similarity', color=color2)
    ax2.plot(layers_list, sims, 's--', color=color2, label='Cosine Similarity')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('Over-smoothing: GCN Depth vs Accuracy and Embedding Similarity')
    plt.tight_layout()
    plt.savefig('plots/ex1_oversmoothing.png', dpi=120)
    plt.close()
    logger.info('Saved plots/ex1_oversmoothing.png')
    return results


#Exercise 2: Model Comparison + Attention Viz
def exercise2(X, Y, A_norm, A_movie, adj_list, train_mask, val_mask, test_mask,
              n_classes, args, logger,
              gcn_emb, gcn_test_acc, gcn_epoch_times,
              gat_emb, gat_test_acc, gat_epoch_times, attn,
              sage_emb, sage_test_acc, sage_epoch_times):
    logger.info('=== Exercise 2: Model Comparison + Attention Visualization ===')

    # Comparison table
    logger.info('Model Comparison:')
    logger.info(f'  GCN       | Test Acc: {gcn_test_acc*100:.2f}% | Avg epoch: {np.mean(gcn_epoch_times)*1000:.0f}ms')
    logger.info(f'  GAT       | Test Acc: {gat_test_acc*100:.2f}% | Avg epoch: {np.mean(gat_epoch_times)*1000:.0f}ms')
    logger.info(f'  GraphSAGE | Test Acc: {sage_test_acc*100:.2f}% | Avg epoch: {np.mean(sage_epoch_times)*1000:.0f}ms')

    # t-SNE comparison plot
    colors_map = plt.cm.tab20(np.linspace(0, 1, n_classes))
    fig, axes  = plt.subplots(1, 3, figsize=(20, 6))
    labels_np  = Y.cpu().numpy()

    for ax, (name, emb, acc) in zip(axes, [
        ('GCN',       gcn_emb,  gcn_test_acc),
        ('GAT',       gat_emb,  gat_test_acc),
        ('GraphSAGE', sage_emb, sage_test_acc),
    ]):
        emb_np = emb.cpu().detach().numpy()
        proj   = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb_np)
        for c in range(n_classes):
            mask = labels_np == c
            if mask.any():
                ax.scatter(proj[mask,0], proj[mask,1],
                           c=[colors_map[c]], label=GENRE_COLS[c], alpha=0.7, s=12)
        ax.set_title(f'{name} (Test: {acc*100:.1f}%)')
        ax.legend(fontsize=5, markerscale=2, ncol=2)
        ax.axis('off')

    plt.suptitle('Movie Genre Embeddings: GCN vs GAT vs GraphSAGE')
    plt.tight_layout()
    plt.savefig('plots/ex2_tsne_comparison.png', dpi=120)
    plt.close()
    logger.info('Saved plots/ex2_tsne_comparison.png')

    # Attention visualization: 3 sample nodes
    attn_np    = attn.cpu().detach().numpy()
    labels_np2 = Y.cpu().numpy()
    sample_nodes = [0, 10, 50]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, node in zip(axes, sample_nodes):
        node_attn  = attn_np[node]
        top5_idx   = np.argsort(node_attn)[-5:][::-1]
        top5_attn  = node_attn[top5_idx]
        node_genre = GENRE_COLS[labels_np2[node]]
        top5_genres = [GENRE_COLS[labels_np2[i]] for i in top5_idx]
        colors_bar  = ['green' if g == node_genre else 'salmon' for g in top5_genres]
        ax.barh(range(5), top5_attn[::-1], color=colors_bar[::-1])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'Node {i}\n({g})' for i, g in
                            zip(top5_idx[::-1], top5_genres[::-1])], fontsize=8)
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'Node {node} ({node_genre})\nGreen = same genre')
    plt.suptitle('GAT Attention: Top-5 Attended Neighbors per Node')
    plt.tight_layout()
    plt.savefig('plots/ex2_attention.png', dpi=120)
    plt.close()
    logger.info('Saved plots/ex2_attention.png')


#Exercise 3: MLP Baseline
def exercise3(X, Y, train_mask, val_mask, test_mask, n_classes, args, logger,
              gcn_test_acc, gat_test_acc, sage_test_acc):
    logger.info('=== Exercise 3: MLP Baseline ===')
    _, mlp_test_acc, _, _ = train_mlp(
        X, Y, train_mask, val_mask, test_mask, n_classes, args, logger)

    logger.info('Final Comparison:')
    logger.info(f'  MLP (no graph) | Test Acc: {mlp_test_acc*100:.2f}%')
    logger.info(f'  GCN            | Test Acc: {gcn_test_acc*100:.2f}%')
    logger.info(f'  GAT            | Test Acc: {gat_test_acc*100:.2f}%')
    logger.info(f'  GraphSAGE      | Test Acc: {sage_test_acc*100:.2f}%')
    logger.info(f'  Best GNN gain over MLP: {(max(gcn_test_acc, sage_test_acc) - mlp_test_acc)*100:.2f}pp')
    return mlp_test_acc

#Main
def main():
    args   = parse_args()
    set_seed(args.seed)
    logger = get_logger('a5')
    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    os.makedirs('plots', exist_ok=True)

    # Load data
    (X, Y, A_movie, A_coo, adj_list,
     n_classes, N_MOVIES, movie_ids, mid2idx,
     train_mask, val_mask, test_mask,
     ratings, movies) = load_movielens(logger)

    A_norm = normalize_adjacency(A_movie)

    run_all = args.all_ex or args.model == 'all'

    # Train base models
    gcn_emb = gat_emb = sage_emb = None
    gcn_test_acc = gat_test_acc = sage_test_acc = 0.0
    gcn_epoch_times = gat_epoch_times = sage_epoch_times = [0]
    attn = None

    if args.model in ('gcn', 'all') or run_all or args.ex1 or args.ex2 or args.ex3:
        logger.info('Training GCN...')
        _, gcn_emb, gcn_test_acc, _, _, gcn_epoch_times = train_gcn(
            X, Y, A_norm, train_mask, val_mask, test_mask,
            n_classes, args, logger, n_layers=2, tag='gcn')

    if args.model in ('gat', 'all') or run_all or args.ex2:
        logger.info('Training GAT...')
        _, gat_emb, attn, gat_test_acc, _, gat_epoch_times = train_gat(
            X, Y, A_movie, train_mask, val_mask, test_mask,
            n_classes, args, logger)

    if args.model in ('sage', 'all') or run_all or args.ex2 or args.ex3:
        logger.info('Training GraphSAGE...')
        _, sage_emb, sage_test_acc, _, _, sage_epoch_times = train_sage(
            X, Y, adj_list, train_mask, val_mask, test_mask,
            n_classes, args, logger)

    # Exercises
    if args.ex1 or run_all:
        exercise1(X, Y, A_norm, train_mask, val_mask, test_mask,
                  n_classes, args, logger)

    if args.ex2 or run_all:
        if attn is None:
            logger.info('Training GAT for Exercise 2...')
            _, gat_emb, attn, gat_test_acc, _, gat_epoch_times = train_gat(
                X, Y, A_movie, train_mask, val_mask, test_mask,
                n_classes, args, logger)
        exercise2(X, Y, A_norm, A_movie, adj_list,
                  train_mask, val_mask, test_mask, n_classes, args, logger,
                  gcn_emb, gcn_test_acc, gcn_epoch_times,
                  gat_emb, gat_test_acc, gat_epoch_times, attn,
                  sage_emb, sage_test_acc, sage_epoch_times)

    if args.ex3 or run_all:
        exercise3(X, Y, train_mask, val_mask, test_mask, n_classes, args, logger,
                  gcn_test_acc, gat_test_acc, sage_test_acc)

    logger.info('Done.')

if __name__ == '__main__':
    main()
