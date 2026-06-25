#!/usr/bin/env python3
"""
Cross-phrasing transfer analysis (local; answers VLAF point 1).

For each (train_variant, test_variant): build the monitored-vs-unmonitored
direction on train_variant's layer-15 activations, then test whether it still
separates the conditions in test_variant. Off-diagonal high => the direction
encodes the abstract monitoring STATE (transfers across surface form). Collapse
to ~0.5 => it is specific to the surface form (token-bound).

Primary method: contrastive mean-diff (z-scored, last-token) — matches the
direction SIEVE originally audited. Secondary: logistic regression (more
powerful). Control: TF-IDF on the full prompt text (surface-token baseline).
Diagonal cells use leave-one-family-out so train!=test there too.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

ACTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vlaf_acts"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
LAYER = 15
VARIANTS = ["v0_tag", "v1_paraphrase", "v2_renamed", "v3_implicit"]


def load_variant(v):
    d = np.load(os.path.join(ACTS_DIR, f"{v}_acts.npz"), allow_pickle=True)
    return d["acts"], d["labels"].astype(int), d["family"].astype(str), d["pid"].astype(str)


def zscore_fit(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return mu, sd


def meandiff_auroc(Xtr, ytr, Xte, yte):
    mu, sd = zscore_fit(Xtr)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Ztr[ytr == 0].mean(0) - Ztr[ytr == 1].mean(0)   # monitored(0) - unmonitored(1)
    s = Zte @ d
    return roc_auc_score((yte == 0).astype(int), s)


def logreg_auroc(Xtr, ytr, Xte, yte):
    mu, sd = zscore_fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((Xtr - mu) / sd, (ytr == 0).astype(int))
    p = clf.decision_function((Xte - mu) / sd)
    return roc_auc_score((yte == 0).astype(int), p)


def lofo(method, X, y, fam):
    aucs = []
    for f in np.unique(fam):
        tr, te = fam != f, fam == f
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(method(X[tr], y[tr], X[te], y[te]))
    return float(np.mean(aucs))


def matrix(method, data):
    M = np.zeros((4, 4))
    for i, tv in enumerate(VARIANTS):
        Xtr, ytr, ftr, _ = data[tv]
        for j, sv in enumerate(VARIANTS):
            Xte, yte, fte, _ = data[sv]
            if i == j:
                M[i, j] = lofo(method, Xtr, ytr, ftr)            # fair within-variant
            else:
                M[i, j] = method(Xtr, ytr, Xte, yte)             # train all X, test all Y
    return M


def print_matrix(name, M):
    print(f"\n=== {name} (rows=train phrasing, cols=test phrasing) ===")
    print("            " + "".join(f"{v[:11]:>13}" for v in VARIANTS))
    for i, v in enumerate(VARIANTS):
        print(f"{v:>11} " + "".join(f"{M[i,j]:13.3f}" for j in range(4)))


def main():
    files = glob.glob(os.path.join(ACTS_DIR, "*_acts.npz"))
    print(f"[transfer] acts dir={ACTS_DIR}; files={[os.path.basename(f) for f in files]}")
    data = {}
    for v in VARIANTS:
        acts, y, fam, pid = load_variant(v)
        print(f"[transfer] {v}: acts{acts.shape} n={len(y)} monitored={int((y==0).sum())} unmonitored={int((y==1).sum())}")
        data[v] = (acts[:, LAYER, :].astype(np.float32), y, fam, pid)

    md = matrix(meandiff_auroc, data)
    lr = matrix(logreg_auroc, data)
    print_matrix(f"CONTRASTIVE MEAN-DIFF direction, layer {LAYER} (PRIMARY)", md)
    print_matrix(f"LOGISTIC probe, layer {LAYER} (secondary)", lr)

    # --- TF-IDF surface-text control (transfer of a text classifier) ---
    texts = {}
    for v in VARIANTS:
        pid2txt = {}
        for line in open(os.path.join(PROMPTS_DIR, f"{v}.jsonl")):
            r = json.loads(line)
            pid2txt[r["prompt_id"]] = r["text"]
        _, y, fam, pid = data[v][1], *data[v][1:]
        texts[v] = ([pid2txt[p] for p in data[v][3]], data[v][1])
    Mtf = np.zeros((4, 4))
    for i, tv in enumerate(VARIANTS):
        Xtr_txt, ytr = texts[tv]
        vec = TfidfVectorizer(max_features=3000).fit(Xtr_txt)
        clf = LogisticRegression(max_iter=2000).fit(vec.transform(Xtr_txt), (ytr == 0).astype(int))
        for j, sv in enumerate(VARIANTS):
            Xte_txt, yte = texts[sv]
            p = clf.decision_function(vec.transform(Xte_txt))
            Mtf[i, j] = roc_auc_score((yte == 0).astype(int), p)
    print_matrix("TF-IDF surface-text classifier (control)", Mtf)

    # --- verdict ---
    decisive = [("v0_tag->v2_renamed", md[0, 2]), ("v0_tag->v3_implicit", md[0, 3]),
                ("v2_renamed->v0_tag", md[2, 0]), ("v3_implicit->v0_tag", md[3, 0]),
                ("v2_renamed->v3_implicit", md[2, 3]), ("v3_implicit->v2_renamed", md[3, 2])]
    tf_decisive = np.mean([Mtf[0, 2], Mtf[0, 3], Mtf[2, 0], Mtf[3, 0]])
    md_decisive = np.mean([v for _, v in decisive])
    print("\n=== DECISIVE CELLS (no shared 'free-tier' vocabulary) ===")
    for k, v in decisive:
        print(f"  {k:28} mean-diff AUROC = {v:.3f}")
    print(f"\n  mean of decisive probe cells : {md_decisive:.3f}")
    print(f"  mean of same TF-IDF cells    : {tf_decisive:.3f}")
    print("\n=== READING ===")
    print("  diagonal high + decisive cells high  -> direction encodes the MONITORING STATE")
    print("     (transfers across surface form); the 'surface-token artifact' reading is WRONG.")
    print("  decisive cells ~0.5                  -> direction is surface-form specific.")
    print("  probe decisive >> TF-IDF decisive    -> probe captures structure beyond surface text.")


if __name__ == "__main__":
    main()
