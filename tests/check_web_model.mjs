// Checks the browser model (web/model.js) against probabilities from the Python pipeline.
// Run from the project root: python scripts/export_web_model.py && node tests/check_web_model.mjs
import { readFileSync } from "node:fs";
import { predictProba } from "../web/model.js";

const model = JSON.parse(readFileSync(new URL("../web/model.json", import.meta.url)));
const { texts, proba } = JSON.parse(readFileSync(new URL("./web_expected.json", import.meta.url)));

let worst = 0;
texts.forEach((text, i) => {
  const got = predictProba(model, text).proba;
  got.forEach((p, k) => (worst = Math.max(worst, Math.abs(p - proba[i][k]))));
});
console.log(`${texts.length} cases, max probability difference ${worst.toExponential(2)}`);
if (worst > 1e-9) {
  console.error("FAIL: browser model does not match the Python pipeline");
  process.exit(1);
}
console.log("OK");
