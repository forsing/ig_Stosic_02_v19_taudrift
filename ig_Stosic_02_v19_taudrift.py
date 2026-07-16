from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
p_froz vs p_move lift + overdue; smer zavisi od frozen_now.

τ ≈ 0 drift (paper 1: zamrznut koordinat, zero drift).

  χ_t = log((cold+ε)/(warm+ε))
  τ_t = rolling mean χ over L
  drift_t = τ_t − τ_{t−1}
  frozen_t = |drift_t| ≤ δ   (δ = p20 |drift|)

  score = ±(p_froz − p_move)/√p_all + overdue
Ban last; next. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from math import log
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
EPS = 1e-6
L_TAU = 20
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def chi_of(draw) -> float:
    cold = sum(1 for x in draw if int(x) <= 13)
    warm = sum(1 for x in draw if int(x) >= 27)
    return log((cold + EPS) / (warm + EPS))


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def regime_rates(
    draws: np.ndarray, frozen_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """presence rates u frozen / move / all (po kolu)."""
    n = len(draws)
    pf = np.zeros(FRONT_N)
    pm = np.zeros(FRONT_N)
    pa = np.zeros(FRONT_N)
    nf = float(frozen_mask.sum()) + 1e-12
    nm = float((~frozen_mask).sum()) + 1e-12
    for t in range(n):
        for x in draws[t].tolist():
            i = int(x) - 1
            pa[i] += 1.0
            if frozen_mask[t]:
                pf[i] += 1.0
            else:
                pm[i] += 1.0
    pf /= nf
    pm /= nm
    pa /= float(n)
    return pf, pm, pa


def overdue_vec(draws: np.ndarray) -> np.ndarray:
    present = np.zeros((len(draws), FRONT_N), dtype=float)
    for i, d in enumerate(draws):
        for x in d.tolist():
            present[i, int(x) - 1] = 1.0
    t_end = len(draws) - 1
    ov = np.zeros(FRONT_N)
    for n in range(FRONT_N):
        idx = np.where(present[:, n] == 1.0)[0]
        mean_g = float(np.diff(idx).mean()) if len(idx) >= 2 else float(len(draws))
        last = int(idx[-1]) if len(idx) else -1
        cur = t_end - last if last >= 0 else len(draws)
        ov[n] = cur / mean_g if mean_g > 0 else 0.0
    return ov


def number_scores(
    pf: np.ndarray,
    pm: np.ndarray,
    pa: np.ndarray,
    ov: np.ndarray,
    frozen_now: bool,
    ban: set[int],
) -> dict[int, float]:
    """
    lift = p_froz - p_move, Fisher-težina 1/sqrt(p_all).
    frozen_now → +lift; move → −lift (brojevi režima pomeraja).
    """
    max_ov = float(ov.max()) if ov.max() > 0 else 1.0
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
            continue
        w = 1.0 / np.sqrt(max(pa[i], 1e-12))
        lift = float(pf[i] - pm[i]) * w
        s = lift if frozen_now else -lift
        s += 0.35 * float(ov[i] / max_ov)
        out[n] = s
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v19(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    chi = np.array([chi_of(d) for d in draws], dtype=float)
    tau = np.array(
        [chi[max(0, i - L_TAU + 1) : i + 1].mean() for i in range(len(chi))],
        dtype=float,
    )
    drift_series = np.diff(tau, prepend=tau[0])
    abs_d = np.abs(drift_series[L_TAU:])
    delta = float(np.percentile(abs_d, 20)) if len(abs_d) else 1e-6
    frozen_mask = np.abs(drift_series) <= delta
    # ne koristi prvih L_TAU (τ nestabilan)
    frozen_mask[:L_TAU] = False
    drift = float(drift_series[-1])
    frozen_now = bool(frozen_mask[-1])

    pf, pm, pa = regime_rates(draws, frozen_mask)
    ov = overdue_vec(draws)
    score = number_scores(pf, pm, pa, ov, frozen_now, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | L_TAU={L_TAU} | ig_02_v19 τ drift")
    print(f"last: {last.tolist()}")
    print()
    print("=== τ / drift ===")
    print(
        {
            "tau_now": round(float(tau[-1]), 6),
            "drift": round(drift, 6),
            "delta_p20": round(delta, 6),
            "frozen_now": frozen_now,
            "n_frozen": int(frozen_mask.sum()),
            "n_move": int((~frozen_mask).sum()),
        }
    )
    print()
    ranked = sorted(
        ((n, float(score[n])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (p_froz−p_move) ===")
    print([(n, round(sc, 6)) for n, sc in ranked[:12]])
    print()
    print("=== next (ig_02_v19 τ) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_02_v19()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | L_TAU=20 | ig_02_v19 τ drift
last: [4, 5, 6, 11, 12, 18, 28]

=== τ / drift ===
{'tau_now': 1.672573, 'drift': 0.115129, 'delta_p20': 0.020273, 'frozen_now': False, 'n_frozen': 927, 'n_move': 3723}

=== top12 skor (p_froz−p_move) ===
[(21, 0.415024), (2, 0.329105), (26, 0.295027), (25, 0.277032), (30, 0.274723), (38, 0.240191), (23, 0.233973), (19, 0.198881), (33, 0.190762), (35, 0.183299), (14, 0.171298), (9, 0.159232)]

=== next (ig_02_v19 τ) ===
next: [2, 14, 21, 23, 25, 26, 30]
"""
