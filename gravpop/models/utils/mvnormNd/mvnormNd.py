import numpy as np
import scipy
from dataclasses import dataclass
from typing import Sequence, Tuple, List

from scipy.stats import norm, multivariate_normal
# mvnun is deprecated but still available; it computes P(a < X < b) directly.
# If it's unavailable in your SciPy, switch to mvn.mvnun import or use a Genz wrapper.
try:
    from scipy.stats.mvn import mvnun
except Exception:
    mvnun = None  # we'll fall back to multivariate_normal.cdf on small dims (slow)

# ------------------------- utilities -------------------------

def _complement_indices(d: int, idx: Sequence[int]) -> np.ndarray:
    """Return sorted complement of idx in 0..d-1."""
    mask = np.ones(d, dtype=bool)
    mask[np.asarray(idx, dtype=int)] = False
    return np.nonzero(mask)[0]

def _safe_logdiffexp(a: float, b: float) -> float:
    """Compute log(exp(a) - exp(b)) stably, assuming a >= b."""
    if a < b:
        # For our use (logcdf(b) - logcdf(a)) we expect a >= b; guard anyway.
        a, b = b, a
        sign = -1.0
    else:
        sign = 1.0
    # log(exp(a) - exp(b)) = a + log1p(-exp(b-a))
    return a + np.log1p(-np.exp(b - a)) * sign

def _mv_box_prob(mu: np.ndarray, Sigma: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute P(a < Y < b) for Y ~ N(mu, Sigma).
    Uses mvnun if available; otherwise falls back to multivariate_normal.cdf
    by splitting into inclusion-exclusion over corners (2^d) – practical only for d<=3–4.
    """
    d = mu.shape[0]
    if d == 0:
        return 1.0
    if d == 1:
        sd = np.sqrt(Sigma[0, 0])
        return norm.cdf((b[0] - mu[0]) / sd) - norm.cdf((a[0] - mu[0]) / sd)
    if mvnun is not None:
        p, info = mvnun(a, b, mu, Sigma)
        # info==0 indicates normal completion.
        return float(p)

    # Fallback (small d): inclusion-exclusion sum over vertices
    # P(a<Y<b)=sum_{S subset {1..d}} (-1)^|S| CDF at x where x_i = b_i if i∉S else a_i
    # This can be slow and numerically rough for larger d.
    total = 0.0
    for mask in range(1 << d):
        x = np.empty(d)
        parity = 0
        for i in range(d):
            if (mask >> i) & 1:
                x[i] = a[i]
                parity ^= 1
            else:
                x[i] = b[i]
        c = multivariate_normal.cdf(x, mean=mu, cov=Sigma)
        total += (-1.0) ** parity * c
    return total

# ------------------------- conditional blocks -------------------------

def mu_notkk(k_idx: Sequence[int], x: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """μ_{-kk} = Σ[-k,k] Σ[k,k]^{-1} x   (k may be a single index or vector of indices)."""
    k_idx = np.atleast_1d(np.array(k_idx, dtype=int))
    d = Sigma.shape[0]
    nk = _complement_indices(d, k_idx)
    S_kk = Sigma[np.ix_(k_idx, k_idx)]
    S_nk_k = Sigma[np.ix_(nk, k_idx)]
    return S_nk_k @ np.linalg.solve(S_kk, x)

def Sigma_notkk(k_idx: Sequence[int], Sigma: np.ndarray) -> np.ndarray:
    """Σ_{-kk} = Σ[-k,-k] - Σ[-k,k] Σ[k,k]^{-1} Σ[k,-k]."""
    k_idx = np.atleast_1d(np.array(k_idx, dtype=int))
    d = Sigma.shape[0]
    nk = _complement_indices(d, k_idx)
    if nk.size == 0:
        return np.zeros((0, 0), dtype=Sigma.dtype)
    S_kk = Sigma[np.ix_(k_idx, k_idx)]
    S_nk_nk = Sigma[np.ix_(nk, nk)]
    S_nk_k = Sigma[np.ix_(nk, k_idx)]
    S_k_nk = Sigma[np.ix_(k_idx, nk)]
    middle = np.linalg.solve(S_kk, S_k_nk)
    return S_nk_nk - S_nk_k @ middle

# ------------------------- core F terms -------------------------

def F_single_k(k: int, x: float, Sigma: np.ndarray,
               a: np.ndarray, b: np.ndarray) -> float:
    """
    F(k, x, Σ, a, b) from your Julia:
    exp( logpdf(N(0, Σ_kk), x) + log P( a_{-k} < Y_{-k} < b_{-k} | X_k = x ) )
    """
    d = Sigma.shape[0]
    nk = _complement_indices(d, [k])
    S_kk = Sigma[k, k]
    logpdf_part = norm.logpdf(x, loc=0.0, scale=np.sqrt(S_kk))

    # No other dims -> conditional probability term is 1 (log=0)
    if nk.size == 0:
        return np.exp(logpdf_part)

    # Conditional mean/cov of X_{-k} | X_k = x
    mu_c = mu_notkk([k], np.array([x]), Sigma)
    S_c = Sigma_notkk([k], Sigma)

    if S_c.size == 1:  # 1x1
        sd = np.sqrt(S_c[0, 0])
        u = (b[nk][0] - mu_c[0]) / sd
        l = (a[nk][0] - mu_c[0]) / sd
        logcdf_b = norm.logcdf(u)
        logcdf_a = norm.logcdf(l)
        logcdf_part = _safe_logdiffexp(logcdf_b, logcdf_a)
        return np.exp(logpdf_part + logcdf_part)

    # General case via mvn box probability
    p_box = _mv_box_prob(mu_c, S_c, a[nk], b[nk])
    return float(np.exp(logpdf_part) * p_box)

def F_multi_kq(kq: Sequence[int], x: np.ndarray, Sigma: np.ndarray,
               a: np.ndarray, b: np.ndarray) -> float:
    """
    F([k,q], x_vec, Σ, a, b) generalization.
    """
    kq = np.atleast_1d(np.array(kq, dtype=int))
    d = Sigma.shape[0]
    nk = _complement_indices(d, kq)

    # logpdf part for joint N(0, Σ[kq,kq]) at x
    S_kq_kq = Sigma[np.ix_(kq, kq)]
    logpdf_part = multivariate_normal.logpdf(x, mean=np.zeros(len(kq)), cov=S_kq_kq)

    if nk.size == 0:
        return np.exp(logpdf_part)

    mu_c = mu_notkk(kq, np.asarray(x), Sigma)
    S_c = Sigma_notkk(kq, Sigma)

    if S_c.size == 1:
        sd = np.sqrt(S_c[0, 0])
        u = (b[nk][0] - mu_c[0]) / sd
        l = (a[nk][0] - mu_c[0]) / sd
        logcdf_part = _safe_logdiffexp(norm.logcdf(u), norm.logcdf(l))
        return np.exp(logpdf_part + logcdf_part)

    p_box = _mv_box_prob(mu_c, S_c, a[nk], b[nk])
    return float(np.exp(logpdf_part) * p_box)

# ------------------------- α, E[X], E[XXᵀ] -------------------------

def alpha(Sigma: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """α = P(a < X < b) for X ~ N(0, Σ)."""
    d = Sigma.shape[0]
    if d == 1:
        sd = np.sqrt(Sigma[0, 0])
        return norm.cdf(b[0] / sd) - norm.cdf(a[0] / sd)
    return _mv_box_prob(np.zeros(d), Sigma, a, b)

def EX_i(Sigma: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    E[X] for the zero-mean truncated box, where X ~ N(0, Σ) | a < X < b.
    Implements Σ * [F(k,a_k) - F(k,b_k)] / α
    """
    d = Sigma.shape[0]
    vec = np.empty(d, dtype=float)
    α = alpha(Sigma, a, b)
    for k in range(d):
        vec[k] = (F_single_k(k, a[k], Sigma, a, b) -
                  F_single_k(k, b[k], Sigma, a, b)) / α
    return Sigma @ vec

def EX_i_inplace(out_M1: np.ndarray, tmp: np.ndarray,
                 Sigma: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    In-place version of EX_i: writes result to out_M1, uses tmp as scratch.
    """
    d = Sigma.shape[0]
    α = alpha(Sigma, a, b)
    for k in range(d):
        tmp[k] = (F_single_k(k, a[k], Sigma, a, b) -
                  F_single_k(k, b[k], Sigma, a, b)) / α
    # out_M1 = Σ * tmp
    np.dot(Sigma, tmp, out=out_M1)

def EX_iX_j_inplace(out_M2: np.ndarray, Sigma: np.ndarray,
                    a: np.ndarray, b: np.ndarray):
    """
    In-place E[XXᵀ] for the zero-mean truncated distribution,
    following your nested-loop structure.
    """
    d = Sigma.shape[0]
    α0 = alpha(Sigma, a, b)
    # Start with α0 * Σ
    out_M2[:, :] = α0 * Sigma

    for i in range(d):
        for j in range(d):
            acc = out_M2[i, j]  # current value α0 Σ[i,j]
            for k in range(d):
                # con = Σ[i,k] Σ[j,k] / Σ[k,k] * ( a_k F(k,a_k) - b_k F(k,b_k) )
                S_ik = Sigma[i, k]
                S_jk = Sigma[j, k]
                S_kk = Sigma[k, k]
                con = (S_ik * S_jk / S_kk) * (
                    a[k] * F_single_k(k, a[k], Sigma, a, b)
                    - b[k] * F_single_k(k, b[k], Sigma, a, b)
                )

                insidek = 0.0
                for q in range(d):
                    if q == k:
                        continue
                    # con1 = Σ[j,q] - Σ[k,q] Σ[j,k] / Σ[k,k]
                    con1 = Sigma[j, q] - (Sigma[k, q] * S_jk / S_kk)

                    # con2 = F([k,q],[a_k,a_q]) + F([k,q],[b_k,b_q])
                    #      - F([k,q],[a_k,b_q]) - F([k,q],[b_k,a_q])
                    con2 = 0.0
                    con2 += F_multi_kq([k, q], np.array([a[k], a[q]]), Sigma, a, b)
                    con2 += F_multi_kq([k, q], np.array([b[k], b[q]]), Sigma, a, b)
                    con2 -= F_multi_kq([k, q], np.array([a[k], b[q]]), Sigma, a, b)
                    con2 -= F_multi_kq([k, q], np.array([b[k], a[q]]), Sigma, a, b)

                    insidek += con1 * con2

                acc += con + (S_ik * insidek)
            out_M2[i, j] = acc

    out_M2 /= α0

def EX_iX_j(Sigma: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    M2 = np.zeros_like(Sigma, dtype=float)
    EX_iX_j_inplace(M2, Sigma, a, b)
    return M2

# ------------------------- outward-facing API -------------------------

@dataclass
class TruncatedMvNormal:
    mu: np.ndarray          # mean vector μ
    Sigma: np.ndarray       # covariance Σ
    a: np.ndarray           # lower bounds
    b: np.ndarray           # upper bounds

def EY_and_EYYt(tn: TruncatedMvNormal) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (E[Y], E[YYᵀ]) for Y ~ N(μ, Σ) truncated to a < Y < b.
    """
    μ = np.asarray(tn.mu)
    Σ = np.asarray(tn.Sigma)
    a = np.asarray(tn.a) - μ
    b = np.asarray(tn.b) - μ

    EX1 = EX_i(Σ, a, b)                            # E[X] for centered truncated
    EX2 = EX_iX_j(Σ, a, b)                         # E[XXᵀ] for centered truncated

    EY1 = EX1 + μ
    EY2 = EX2 + np.outer(μ, EX1) + np.outer(EX1, μ) + np.outer(μ, μ)
    return EY1, EY2

# ------------------------- block-structure helpers -------------------------

@dataclass
class CovarianceBlockStructure:
    block_indices: List[np.ndarray]   # e.g., [np.array([0,1]), np.array([2,3,4])]

def EX_blocks_inplace(M1: np.ndarray, tmp: np.ndarray,
                      Sigma: np.ndarray, a: np.ndarray, b: np.ndarray,
                      bs: CovarianceBlockStructure):
    # Compute blockwise first moments for centered case.
    for idx in bs.block_indices:
        Σ_ii = Sigma[np.ix_(idx, idx)]
        a_i = a[idx]
        b_i = b[idx]
        # write into M1[idx]
        EX_i_inplace(M1[idx], tmp[idx], Σ_ii, a_i, b_i)

def EXXXT_blocks_inplace(M1: np.ndarray, M2: np.ndarray, tmp: np.ndarray,
                         Sigma: np.ndarray, a: np.ndarray, b: np.ndarray,
                         bs: CovarianceBlockStructure):
    # First pass: within-block E[X]
    EX_blocks_inplace(M1, tmp, Sigma, a, b, bs)
    # Second pass: blockwise E[XXᵀ]
    for i_s, I in enumerate(bs.block_indices):
        Σ_ii = Sigma[np.ix_(I, I)]
        a_i = a[I]; b_i = b[I]
        # within-block exact
        EX_iX_j_inplace(M2[np.ix_(I, I)], Σ_ii, a_i, b_i)

        # off-block: outer product of first moments
        for j_s, J in enumerate(bs.block_indices):
            if j_s == i_s:
                continue
            # Fill M2[I, J] = M1[I] * M1[J]^T and symmetric counterpart
            M2[np.ix_(I, J)] = np.outer(M1[I], M1[J])
            M2[np.ix_(J, I)] = np.outer(M1[J], M1[I])

def EY_EYYt_blocks(mu: np.ndarray, Sigma: np.ndarray,
                   a: np.ndarray, b: np.ndarray,
                   bs: CovarianceBlockStructure) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-structured variant: off-diagonal blocks use M1_i M1_jᵀ as in your Julia.
    """
    μ = np.asarray(mu); Σ = np.asarray(Sigma)
    a0 = np.asarray(a) - μ
    b0 = np.asarray(b) - μ

    d = Σ.shape[0]
    M1 = np.zeros(d, dtype=float)
    M2 = np.zeros((d, d), dtype=float)
    tmp = np.zeros(d, dtype=float)

    EXXXT_blocks_inplace(M1, M2, tmp, Σ, a0, b0, bs)

    EY1 = M1 + μ
    EY2 = M2 + np.outer(μ, M1) + np.outer(M1, μ) + np.outer(μ, μ)
    return EY1, EY2



import jax
import jax.numpy as jnp
import scipy

def cdf_below(x, mu, Sigma):
    return scipy.stats.multivariate_normal.cdf(x, mean=mu, cov=Sigma)

def cdf_above(x, mu, Sigma):
    return scipy.stats.multivariate_normal.cdf(-x, mean=-mu, cov=Sigma)

def mvn_old(mu, Sigma, a, b):
    vals = _mv_box_prob(mu, Sigma,a,b)
    return vals

@jax.custom_vjp
def mvnorm_Nd(mu, Sigma, a, b):
    vals = _mv_box_prob(mu, Sigma,a,b)
    return vals
    
def mvnorm_Nd_fwd(mu, Sigma, a, b):
    alpha = _mv_box_prob(mu, Sigma,a,b)
    return alpha, (alpha, mu, Sigma, a, b)

def mvnorm_Nd_bwd(res, dg_dalpha):
    alpha, mu, Sigma, a, b = res
    
    aminusmu = a - mu;
    bminusmu = b - mu;
    EX1 = EX_i(Sigma, aminusmu, bminusmu)                            # E[X] for centered truncated
    EX2 = EX_iX_j(Sigma, aminusmu, bminusmu)                         # E[XXᵀ] for centered truncated

    invSigma = jnp.linalg.inv(Sigma);

    dalpha_dmu = alpha * invSigma @ EX1;

    dalpha_dSigma = alpha * ( -0.5 * invSigma + 0.5 * invSigma @ EX2 @ invSigma)

    return (dg_dalpha * dalpha_dmu, dg_dalpha * dalpha_dSigma, jnp.zeros_like(a), jnp.zeros_like(b))

mvnorm_Nd.defvjp(mvn_fwd, mvn_bwd)