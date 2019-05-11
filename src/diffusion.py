import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
np.random.seed(27)

def get_adjacency_size(H, W):
    """
    Each of the H x W nodes participates 
    in two edges (amortized). Lose (H + W) 
    edges to boundaries.
    """
    m = H * W
    n = H * W * 2 - (H + W)
    return m, n

def make_hori_conn(i, j):
    """Connect (i, j) with (i, j-1)."""
    H, W = i.shape
    src = (i[:, 1:] * W + j[:, 1:]).ravel()
    dst = (i[:, 1:] * W + j[:, 1:] - 1).ravel()
    rng = np.arange(H * W - W, 2 * H * W - W - H).ravel()
    return src, dst, rng

def make_vert_conn(i, j):
    """Connect (i, j) with (i-1, j)."""
    H, W = i.shape
    src = (i[1:, :] * W + j[1:, :]).ravel()
    dst = ((i[1:, :] - 1) * W + j[1:, :]).ravel()
    rng = np.arange(0, H * W - W).ravel()
    return src, dst, rng

def make_conn(H, W):
    """Connect each node with its
    neighbors up and to left."""
    i, j = np.mgrid[:H, :W]
    h_src, h_dst, h_rng = make_hori_conn(i, j)
    v_src, v_dst, v_rng = make_vert_conn(i, j)
    src = np.r_[h_src, v_src]
    dst = np.r_[h_dst, v_dst]
    rng = np.r_[h_rng, v_rng]
    return src, dst, rng

def init_sparse_adjacency(H, W):
    """Use lil_matrix for efficient sparse update."""
    m, n = get_adjacency_size(H, W)
    A = scipy.sparse.lil_matrix((m, n))
    return A
    
def make_adjacency(H, W):
    """
    All cols sum to zero and
    have two nonzero entries.
    """
    A = init_sparse_adjacency(H, W)
    src, dst, rng = make_conn(H, W)
    A[src, rng] = -1
    A[dst, rng] = +1
    return A.tocsr()   

def make_resistance(n, rho=1):
    """
    For simplicity assume resistance
    matrix multiple of identity.
    """
    R = scipy.sparse.spdiags(np.full(n, rho), 0, n, n)
    return R

def get_slice(L, maxlength, margin=10):
    """
    Apply magic heuristic to get interesting 
    non-degenerate demo. Best not to ask.
    """
    v0 = np.random.randint(L // margin, L)
    v1 = v0 + np.random.randint(min(L - v0, maxlength))
    return slice(v0, v1)   

def get_fixed_node_mask(H, W, k=3, maxwidth=30, maxheight=10):
    """
    True where node potential held fixed.
    """
    mask = np.full((H, W), False)
    for _ in range(k):
        r = get_slice(H, maxheight)
        c = get_slice(W, maxwidth)
        mask[r, c] = True
    return mask

def masked_scatter(size, val, mask, default=0):
    """
    Combined fill and masked scatter.
    """
    arr = np.full(size, default)
    arr[mask] = val
    return arr

def make_fixed_node_mats(H, W, fixed_val=1):
    """
    Matrices B and C enforce fixed node 
    and source/sink constraints.
    """
    m, n = get_adjacency_size(H, W)
    mask = get_fixed_node_mask(H, W).ravel()
    B = scipy.sparse.lil_matrix((m, m))
    C = scipy.sparse.lil_matrix((m, m))
    d = masked_scatter(m, fixed_val, mask)
    B[~mask, ~mask] = 1
    C[mask, mask] = 1
    return B, C, d
   
def make_blocksys(A, R, B, C):
    """
    Make block 3x3 system for solving:
        A f + I s +   0 e = 0
        R f + 0 s + A.T e = 0
        0 f + B s +   C e = d
    In the stacked variable (f, s, e).
    """
    m, n = A.shape
    I_mxm = scipy.sparse.eye(m)
    Z_mxm = scipy.sparse.csr_matrix((m, m))
    Z_nxm = scipy.sparse.csr_matrix((n, m))
    Z_mxn = scipy.sparse.csr_matrix((m, n))
    blocksys = scipy.sparse.bmat([
        [A, I_mxm, Z_mxm],
        [R, Z_nxm, A.T],
        [Z_mxn, B, C],
    ]).tocsr()
    return blocksys

def make_problem(H, W):
    """
    Construct problem instance for
    diffusion in grid network.
    """
    m, n = get_adjacency_size(H, W)
    A = make_adjacency(H, W)
    R = make_resistance(n)
    B, C, d = make_fixed_node_mats(H, W)
    blocksys = make_blocksys(A, R, B, C)
    rhs = np.r_[np.zeros(m), np.zeros(n), d]
    return blocksys, rhs

def how_sparse(mat):
    """Summarize problem size and structure."""
    m, n = mat.shape
    pct = mat.nnz / (m * n)
    msg = f'Linear system: {m} eqns in {n} vars.\n'
    msg += f'Sparsity: {pct:.010f}.'
    print(msg)
    
def viz_diffusion(node_potentials, H, W):
    """Plot realized node potential as grid."""
    img = node_potentials.reshape(H, W)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.savefig('../images/diffusion.png', 
        bbox_inches='tight', 
        pad_inches=0,
        dpi=200,
    )

def main():
    """
    Can comfortably push to 500x500 
    on intel i7 w/ 8GB RAM.
    """
    H, W = 500, 500
    m, n = get_adjacency_size(H, W)
    blocksys, rhs = make_problem(H, W)
    fse = scipy.sparse.linalg.spsolve(blocksys, rhs)
    f, s, e = np.split(fse, [m, m + n])   
    how_sparse(blocksys)
    viz_diffusion(e, H, W)
    
if __name__ == '__main__':
    main()
