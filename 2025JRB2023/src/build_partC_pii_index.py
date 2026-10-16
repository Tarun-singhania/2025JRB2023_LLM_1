#!/usr/bin/env python3
"""Build the local Part-C PII index used by untargeted extraction.

The assignment CLI references data/partC_pii_index.jsonl, but that file was not
present in the supplied bundle used for these experiments. This script builds a
documented detector-derived index from one raw text per distinct doc_id using
Regex + Presidio. It does not print detected values.
"""
import argparse, json, re
from pathlib import Path
from collections import Counter


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().casefold()


REGEXES = {
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "URL": re.compile(r"\bhttps?://[^\s<>\]\)]+|\bwww\.[^\s<>\]\)]+", re.I),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PHONE_NUMBER": re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "DATE_TIME": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/partC_corpus_raw.jsonl")
    ap.add_argument("--output_file", default="data/partC_pii_index.jsonl")
    ap.add_argument("--audit_file", default="results/partC_pii_index_build.json")
    args = ap.parse_args()

    rows = read_jsonl(args.corpus)
    docs = {}
    for r in rows:
        did = str(r["doc_id"])
        txt = str(r["text"])
        if did in docs and docs[did] != txt:
            raise RuntimeError(f"Repeated doc_id has inconsistent text: {did}")
        docs.setdefault(did, txt)

    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
    except Exception as e:
        raise RuntimeError(
            "Presidio could not initialize. Install presidio-analyzer and its NLP dependencies."
        ) from e

    output_rows = []
    label_counts = Counter()
    distinct_values = set()

    for doc_id, text in docs.items():
        found = {}
        for label, pat in REGEXES.items():
            for m in pat.finditer(text):
                value = m.group(0)
                key = (norm(value), label)
                if not key[0]:
                    continue
                item = found.setdefault(key, {"value": value, "label": label, "sources": set()})
                item["sources"].add("regex")

        for r in analyzer.analyze(text=text, language="en"):
            value = text[int(r.start):int(r.end)]
            label = str(r.entity_type)
            key = (norm(value), label)
            if not key[0]:
                continue
            item = found.setdefault(key, {"value": value, "label": label, "sources": set()})
            item["sources"].add("presidio")

        pii_values = []
        for (_nv, _label), item in sorted(found.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            pii_values.append({
                "value": item["value"],
                "label": item["label"],
                "source": "+".join(sorted(item["sources"])),
            })
            label_counts[item["label"]] += 1
            distinct_values.add(norm(item["value"]))

        output_rows.append({"doc_id": doc_id, "pii_values": pii_values})

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    audit = {
        "construction": "union of local regex and Presidio detections on one raw text per distinct doc_id",
        "n_docs": len(output_rows),
        "n_docs_with_detected_pii": sum(bool(x["pii_values"]) for x in output_rows),
        "n_detected_entries": sum(len(x["pii_values"]) for x in output_rows),
        "n_distinct_normalized_values": len(distinct_values),
        "by_label": dict(sorted(label_counts.items())),
        "limitation": "detector-derived index; recovery rate therefore inherits detector false positives and false negatives",
    }
    audit_path = Path(args.audit_file)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("PART-C PII INDEX BUILD: PASS")
    print("documents:", audit["n_docs"])
    print("docs with detected PII:", audit["n_docs_with_detected_pii"])
    print("detected entries:", audit["n_detected_entries"])
    print("values are intentionally not printed")


if __name__ == "__main__":
    main()
