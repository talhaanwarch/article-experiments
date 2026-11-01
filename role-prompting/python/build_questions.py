"""Build the question sets from Zheng et al. (EMNLP Findings 2024) at a pinned commit.

The paper's repository ships 2,457 MMLU questions in data/mmlu_sample_ques.csv.
Its plot notebook (analysis_notebooks/plot.ipynb) formats each one as
"{question} Options: 1. a, 2. b, 3. c, 4. d", drops duplicate texts and keeps
2,410, the number the paper reports. The same notebook maps the 26 MMLU
subjects to 8 domains (domain_dic). Both steps are copied here.

Outputs:
  ../data/questions_all.csv   all 2,410 questions (used for the rebuild check)
  ../data/questions.csv       the 300-question sample: 37 or 38 per domain,
                              drawn with random.Random(SEED)
"""

import csv
import io
import random
from collections import defaultdict
from pathlib import Path

import httpx

COMMIT = "9f0ab13dcea5260554eab72eb124d4c7557881e1"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Jiaxin-Pei/Prompting-with-Social-Roles/"
    f"{COMMIT}/data/mmlu_sample_ques.csv"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "cache" / "mmlu_sample_ques.csv"
SEED = 767  # the GitHub issue number that tracks this series
SAMPLE_SIZE = 300

# analysis_notebooks/plot.ipynb, domain_dic, word for word.
DOMAINS = {
    "law": ["professional_law", "international_law"],
    "medicine": ["clinical_knowledge", "college_medicine", "professional_medicine"],
    "eecs": ["electrical_engineering", "college_computer_science", "high_school_computer_science"],
    "math": ["high_school_statistics", "college_mathematics", "high_school_mathematics"],
    "politics": ["us_foreign_policy", "high_school_government_and_politics"],
    "psychology": ["professional_psychology", "high_school_psychology"],
    "natural science": ["college_physics", "college_biology", "high_school_physics",
                        "high_school_chemistry", "college_chemistry", "high_school_biology"],
    "econ": ["management", "professional_accounting", "econometrics",
             "high_school_macroeconomics", "high_school_microeconomics"],
}
SUBJECT_DOMAIN = {s: d for d, subjects in DOMAINS.items() for s in subjects}


def format_choices(row: dict) -> str:
    """scripts/vllm_inference_pipeline.py, format_choices."""
    options = [row[f"option{i}"] for i in range(1, 5)]
    choices_text = ", ".join(f"{i}. {option}" for i, option in enumerate(options, 1))
    return f"{row['question']} Options: {choices_text}"


def load_rows() -> list[dict]:
    if not CACHE_FILE.exists():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(SOURCE_URL, timeout=60)
        r.raise_for_status()
        CACHE_FILE.write_bytes(r.content)
    return list(csv.DictReader(io.StringIO(CACHE_FILE.read_text())))


def main() -> None:
    rows, seen = [], set()
    for row in load_rows():
        full = format_choices(row)
        if full in seen:  # ques_df[~ques_df['full_question'].duplicated(keep="first")]
            continue
        seen.add(full)
        rows.append({
            "qid": f"{row['dataset']}/{row['question_id']}",
            "subject": row["dataset"],
            "domain": SUBJECT_DOMAIN[row["dataset"]],
            "true_option": int(row["true_option"]),
            "full_question": full,
        })
    assert len(rows) == 2410, len(rows)
    assert len({r["qid"] for r in rows}) == len(rows)

    fields = ["qid", "subject", "domain", "true_option", "full_question"]
    with (DATA_DIR / "questions_all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    rng = random.Random(SEED)
    domains = sorted(by_domain)
    per_domain = {d: SAMPLE_SIZE // len(domains) for d in domains}
    for d in rng.sample(domains, SAMPLE_SIZE % len(domains)):
        per_domain[d] += 1
    sample = []
    for d in domains:
        pool = sorted(by_domain[d], key=lambda r: r["qid"])
        sample.extend(rng.sample(pool, per_domain[d]))
    with (DATA_DIR / "questions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sample)
    print(f"{len(rows)} questions, sample {len(sample)}: {per_domain}")


if __name__ == "__main__":
    main()
