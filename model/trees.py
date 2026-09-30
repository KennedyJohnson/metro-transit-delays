"""Export LightGBM models as compact JSON and evaluate them exactly like docs/model.js does."""
import numpy as np


def _r(x, n=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _flatten(node, out):
    i = len(out)
    out.append(None)
    if "leaf_index" in node:
        out[i] = [-1, 0, 0, 0, 0, 0, _r(node["leaf_value"])]
        return i
    left, right = _flatten(node["left_child"], out), _flatten(node["right_child"], out)
    mt = {"None": 0, "Zero": 1, "NaN": 2}[node["missing_type"]]
    out[i] = [node["split_feature"], _r(node["threshold"]), int(node["default_left"]), mt, left, right,
              _r(node["internal_value"])]
    return i


def dump(booster, kind: str) -> dict:
    """kind: 'regression' (raw score) or 'binary' (sigmoid)."""
    d = booster.dump_model()
    trees = []
    for t in d["tree_info"]:
        nodes = []
        _flatten(t["tree_structure"], nodes)
        trees.append(nodes)
    return {"kind": kind, "features": d["feature_names"], "trees": trees}


class TreeModel:
    """Vectorized evaluator for dump() output; same node layout and missing-value rules as docs/model.js."""

    def __init__(self, dumped: dict):
        self.kind, self.features = dumped["kind"], dumped["features"]
        self.trees = [np.array([[np.nan if v is None else v for v in n] for n in t], dtype=np.float64)
                      for t in dumped["trees"]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        score = np.zeros(len(X))
        rows = np.arange(len(X))
        for t in self.trees:
            feat, thr, defl, miss = t[:, 0].astype(int), t[:, 1], t[:, 2] == 1, t[:, 3].astype(int)
            left, right, val = t[:, 4].astype(int), t[:, 5].astype(int), t[:, 6]
            node = np.zeros(len(X), dtype=int)
            live = feat[node] != -1
            while live.any():
                r, n = rows[live], node[live]
                x = X[r, feat[n]]
                nan = np.isnan(x)
                nan_default = nan & (miss[n] == 2)
                x = np.where(nan, 0.0, x)
                go_left = np.where(nan_default, defl[n],
                                   np.where((miss[n] == 1) & (np.abs(x) < 1e-35), defl[n], x <= thr[n]))
                node[r] = np.where(go_left, left[n], right[n])
                live = feat[node] != -1
            score += val[node]
        return 1 / (1 + np.exp(-score)) if self.kind == "binary" else score
