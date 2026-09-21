"""Headless tests for the Streamlit dashboard (app.py), using Streamlit's AppTest.

Run from the project folder:   python tests/test_dashboard.py

Checks that every page renders without an exception, that the model loads, that metrics and charts are
displayed, that invalid inputs produce clear errors instead of crashes, and - most importantly - that every
prediction the dashboard shows is identical to the one ``src/predict.py`` returns for the same transaction.
No test trains, tunes or re-evaluates anything; the dataset and the model artefact are only read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import predict as predict_api  # noqa: E402

TIMEOUT = 180
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def render_page(page_function: str, root: str):
    """Script body for AppTest.from_function: import app.py and render one page."""
    import sys as _sys
    _sys.path.insert(0, root)
    import app  # noqa: F401 - defines the pages without starting navigation
    getattr(app, page_function)()


def page_app(page_function: str) -> AppTest:
    return AppTest.from_function(render_page, args=(page_function, str(ROOT)), default_timeout=TIMEOUT)


def texts(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.title] + [e.value for e in at.subheader]
    parts += [e.value for e in at.caption] + [e.value for e in at.info] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


def main() -> int:
    final = json.loads((ROOT / "results" / "phase5_final_metrics.json").read_text(encoding="utf-8"))
    demo = json.loads((ROOT / "results" / "phase5_demo_examples.json").read_text(encoding="utf-8"))
    m = final["metrics"]

    # 1. The full app starts, the navigation is built and the default page renders.
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    check("full app runs without exception", not at.exception, str(at.exception))
    check("default page is Overview", any("Credit Card Fraud Detection" == t.value for t in at.title))
    check("sidebar shows the locked threshold", "0.50" in " ".join(c.value for c in at.sidebar.caption))

    # 2. Every page renders on its own.
    pages = {"page_overview": "Credit Card Fraud Detection", "page_predict": "Fraud Prediction",
             "page_analysis": "Data Analysis", "page_performance": "Model Performance",
             "page_details": "Model Details", "page_about": "About the Project"}
    rendered = {}
    for fn, title in pages.items():
        page = page_app(fn).run()
        rendered[fn] = page
        check(f"{fn} renders without exception", not page.exception, str(page.exception)[:200])
        check(f"{fn} shows its title", any(t.value == title for t in page.title))

    # 3. Metrics come from the Phase 5 results file.
    overview, perf = texts(rendered["page_overview"]), texts(rendered["page_performance"])
    check("overview shows final PR-AUC / precision / recall / F1",
          all(f"{m[k]:.3f}" in overview for k in ("pr_auc", "precision", "recall", "f1")))
    check("performance page shows the final metrics",
          all(f"{m[k]:.4f}" in perf for k in ("pr_auc", "precision", "recall", "f1", "roc_auc", "accuracy")))
    check("performance page shows the confusion counts", all(f"{m[k]}" in perf for k in ("tp", "fn", "fp")))
    check("performance page shows the PR and ROC figures", len(rendered["page_performance"].image) >= 2)
    check("charts render (plotly) on overview, analysis and performance pages",
          all(len(rendered[p].get("plotly_chart")) >= 1 for p in ("page_overview", "page_analysis", "page_performance")))
    check("model details page shows the locked hyperparameters",
          "min_samples_leaf=5" in texts(rendered["page_details"]) and "max_features=0.3" in texts(rendered["page_details"]))

    # 4. Prediction: every demonstration example, through the UI, must equal src/predict.py exactly.
    for ex in demo["examples"]:
        page = page_app("page_predict").run()
        page.button(key=f"ex_{ex['name']}").click().run()
        page.button(key="analyze").click().run()
        shown = page.session_state["last_result"] if "last_result" in page.session_state else None
        backend = predict_api.predict_transaction(ex["features"])
        check(f"prediction '{ex['name']}' - UI equals src/predict.py", shown == backend,
              f"UI {shown} vs backend {backend}")
        check(f"prediction '{ex['name']}' - equals the Phase 5 record",
              shown is not None and abs(shown["fraud_probability"] - ex["fraud_probability"]) <= 1e-12
              and shown["predicted_class"] == ex["predicted_class"])
        verdict = "FRAUD" if backend["predicted_class"] == 1 else "LEGITIMATE"
        check(f"prediction '{ex['name']}' - verdict displayed ({verdict})",
              f"Prediction: {verdict}" in texts(page) and not page.exception)

    # 5. The paste box: a full row pasted as comma-separated values gives the same answer.
    ex = demo["examples"][0]
    page = page_app("page_predict").run()
    page.text_area(key="pasted").input(",".join(repr(ex["features"][c]) for c in predict_api.FEATURE_COLUMNS)).run()
    page.button(key="analyze_paste").click().run()
    shown = page.session_state["last_result"] if "last_result" in page.session_state else None
    check("pasted comma-separated row equals src/predict.py", shown == predict_api.predict_transaction(ex["features"]))

    # 6. Invalid input produces a clear error, never an exception.
    def invalid_case(name: str, setup) -> None:
        page = page_app("page_predict").run()
        setup(page)
        errors = [e.value for e in page.error]
        check(f"invalid input handled - {name}", not page.exception and len(errors) == 1
              and "last_result" not in page.session_state, errors[0][:120] if errors else "no error shown")

    def empty_form(p):
        p.button(key="analyze").click().run()

    def with_value(value):
        def setup(p):
            p.button(key=f"ex_{demo['examples'][2]['name']}").click().run()
            p.text_input(key="field_V7").input(value).run()
            p.button(key="analyze").click().run()
        return setup

    def pasted(text):
        def setup(p):
            p.text_area(key="pasted").input(text).run()
            p.button(key="analyze_paste").click().run()
        return setup

    invalid_case("all fields missing", empty_form)
    invalid_case("malformed number", with_value("12,5x"))
    invalid_case("NaN", with_value("nan"))
    invalid_case("infinity", with_value("inf"))
    invalid_case("empty single field", with_value(""))
    invalid_case("wrong feature count (29 pasted values)", pasted(",".join(["0.1"] * 29)))
    invalid_case("malformed JSON", pasted('{"Time": 0, "V1": '))
    invalid_case("unexpected 'Class' column in JSON",
                 pasted(json.dumps({**{c: 0.0 for c in predict_api.FEATURE_COLUMNS}, "Class": 1})))

    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if (detail and not ok) else ""))
    failed = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} dashboard checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
