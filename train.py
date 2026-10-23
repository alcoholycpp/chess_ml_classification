#!/usr/bin/env python3
import sys, time
import numpy as np, pandas as pd
from sklearn.base import clone

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline


DATA = sys.argv[1] if len(sys.argv) > 1 else "./data/positions.csv"
SEED = 42
THRESH = 1.0                 
TRAIN_SCORE_ROWS = 200_000   
LABELS = ["B", "D", "W"]

df = pd.read_csv(DATA)
y = pd.Series(np.select([df.sf_eval > THRESH, df.sf_eval < -THRESH], ["W", "B"], "D"))
X = df.drop(columns=["game_id", "sf_eval", "result"])       
print(f"{len(X):,} rows, {df.game_id.nunique():,} games, {X.shape[1]} features")
print("missing cells:", int(X.isna().sum().sum()), "| duplicate feature rows:", int(X.duplicated().sum()))

folds = [t for _, t in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, df.game_id)]
te, va, tr = folds[0], folds[1], np.concatenate(folds[2:])
tv = np.concatenate([tr, va])
for name, idx in [("train", tr), ("val", va), ("test", te)]:
    share = y.iloc[idx].value_counts(normalize=True).reindex(LABELS).round(3).to_dict()
    print(f"{name:5s} {len(idx):>8,} rows  {share}")


def scaled(m):
    return make_pipeline(StandardScaler(), m)


CONFIGS = [
    ("Dummy", "most_frequent", "none", DummyClassifier(strategy="most_frequent")),
    ("LogReg", "C=1.0", "StandardScaler", scaled(LogisticRegression(max_iter=1000))),
    *[(f"kNN k={k}", f"k={k}", "StandardScaler", scaled(KNeighborsClassifier(k, n_jobs=-1)))
      for k in (3, 5, 7)],
    ("DecisionTree", "max_depth=8", "none", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
    ("RandomForest", "n_estimators=100, min_samples_leaf=10", "none",
     RandomForestClassifier(100, min_samples_leaf=10, n_jobs=-1, random_state=SEED)),
]


def evaluate(model, idx):
    t = time.time()
    pred = model.predict(X.iloc[idx])
    dt = time.time() - t
    true = y.iloc[idx]
    out = {"accuracy": accuracy_score(true, pred),
           "precision_macro": precision_score(true, pred, average="macro", zero_division=0),
           "recall_macro": recall_score(true, pred, average="macro", zero_division=0),
           "f1_macro": f1_score(true, pred, average="macro")}
    for lab, v in zip(LABELS, f1_score(true, pred, average=None, labels=LABELS, zero_division=0)):
        out[f"f1_{lab}"] = v
    return out, dt, pred


def run(stage, name, params, prep, model, fit_idx, eval_idx, train_sub):
    t = time.time()
    model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    fit_s = time.time() - t
    m, pred_s, pred = evaluate(model, eval_idx)
    train_f1 = f1_score(y.iloc[train_sub], model.predict(X.iloc[train_sub]), average="macro")
    row = {"stage": stage, "model": name, "params": params, "preprocessing": prep,
           "n_fit": len(fit_idx), "n_eval": len(eval_idx), "f1_macro_train": train_f1,
           **m, "fit_s": fit_s, "predict_s": pred_s, "seed": SEED}
    print(f"[{stage}] {name:14s} F1-macro train {train_f1:.4f}  eval {m['f1_macro']:.4f}  "
          f"fit {fit_s:6.1f}s  predict {pred_s:6.1f}s", flush=True)
    return row, pred


rng = np.random.default_rng(SEED)
sub_val = rng.choice(tr, min(TRAIN_SCORE_ROWS, len(tr)), replace=False)
rows = [run("validation", *c, tr, va, sub_val)[0] for c in CONFIGS]

val = pd.DataFrame(rows)
best = val[val.model != "Dummy"].sort_values(["f1_macro", "fit_s"], ascending=[False, True]).model.iloc[0]
print(f"\nselected on validation (max F1-macro, tie -> faster): {best}")

sub_test = rng.choice(tv, min(TRAIN_SCORE_ROWS, len(tv)), replace=False)
for name, params, prep, model in CONFIGS:
    if name in ("Dummy", best):
        row, pred = run("test", name, params, prep, clone(model), tv, te, sub_test)
        rows.append(row)
        if name == best:
            cm = pd.DataFrame(confusion_matrix(y.iloc[te], pred, labels=LABELS),
                              index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
            cm.to_csv("test_confusion.csv")
            print("\nconfusion matrix (test):\n", cm)

res = pd.DataFrame(rows)
res.to_csv("results.csv", index=False)
cols = ["model", "params", "f1_macro_train", "f1_macro", "accuracy", "fit_s", "predict_s"]
print("\nTable 1 - validation (train = fit on train, scored on 50k train subsample)")
print(res[res.stage == "validation"][cols].round(4).to_string(index=False))
print("\nTable 2 - test (fit on train+val)")
print(res[res.stage == "test"][cols[:1] + ["f1_macro", "precision_macro", "recall_macro", "accuracy"]]
      .round(4).to_string(index=False))
print("\nsaved results.csv, test_confusion.csv")
