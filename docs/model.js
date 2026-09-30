// Evaluates the exported LightGBM models in the browser (port of model/trees.py TreeModel).
// Node layout: [feature, threshold, defaultLeft, missingType(0 None,1 Zero,2 NaN), left, right, value]
function goesLeft(node, x) {
  const [, thr, defLeft, missing] = node;
  if (x == null || Number.isNaN(x)) {
    if (missing === 2) return !!defLeft;
    x = 0;
  }
  if (missing === 1 && Math.abs(x) < 1e-35) return !!defLeft;
  return x <= thr;
}

class TreeModel {
  constructor({ kind, features, trees }) {
    this.kind = kind;
    this.features = features;
    this.trees = trees;
  }

  predict(row) {
    const x = this.features.map((f) => (row[f] == null ? NaN : row[f]));
    let score = 0;
    for (const nodes of this.trees) {
      let i = 0;
      while (nodes[i][0] !== -1) i = goesLeft(nodes[i], x[nodes[i][0]]) ? nodes[i][4] : nodes[i][5];
      score += nodes[i][6];
    }
    return this.kind === "binary" ? 1 / (1 + Math.exp(-score)) : score;
  }
}

if (typeof module !== "undefined") module.exports = { TreeModel };
