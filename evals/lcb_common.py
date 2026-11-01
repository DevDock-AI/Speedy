import json, os, random
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
START, END, RELEASE = "2025-02-01", "2025-05-01", "release_v6"
QUOTA, SEED = {"easy": 16, "medium": 21, "hard": 33}, 20261031     # 70 of 131, proportional to the window (31/39/61)
SUBSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_subset_v6_70.json")
def load_window():
    d = load_code_generation_dataset(RELEASE, start_date=START, end_date=END)
    return sorted(d, key=lambda x: x.question_id)
def make_subset():
    d = load_window(); rng = random.Random(SEED); ids = []
    for diff, k in QUOTA.items():
        pool = [x.question_id for x in d if x.difficulty.value == diff]; ids += sorted(rng.sample(pool, k))
    meta = {x.question_id: (x.difficulty.value, str(x.contest_date)[:10], x.platform.value) for x in d}
    json.dump({"release": RELEASE, "start": START, "end": END, "seed": SEED, "quota": QUOTA, "window_size": len(d),
               "problems": [{"id": i, "difficulty": meta[i][0], "date": meta[i][1], "platform": meta[i][2]} for i in ids]},
              open(SUBSET, "w"), indent=1)
def load_subset():
    want = [p["id"] for p in json.load(open(SUBSET))["problems"]]
    by = {x.question_id: x for x in load_window()}
    return [by[i] for i in want]
