// Evaluates docs/features.js + docs/model.js on test cases written by tests/test_model.py.
const fs = require("fs");
const path = require("path");
const { featureRow } = require("../docs/features.js");
const { TreeModel } = require("../docs/model.js");
const [casesFile, dataDir] = process.argv.slice(2);
const { cases, weather, features } = JSON.parse(fs.readFileSync(casesFile));
const rows = cases.map((c) => featureRow({ ...c.args, weather }));
const preds = {};
for (const k of ["median", "late", "early"]) {
  const m = new TreeModel(JSON.parse(fs.readFileSync(path.join(dataDir, `model_${k}.json`))));
  preds[k] = rows.map((r) => m.predict(r));
}
console.log(JSON.stringify({ rows: rows.map((r) => features.map((f) => (Number.isNaN(r[f]) ? null : r[f]))), preds }));
