import json
from collections import Counter
'''
def retrieve_items():
    items = set()

    with open("train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            battle = json.loads(line)
            for mon in battle.get("p1_team_details", []):
                item = mon.get("item")
                if item:
                    items.add(item)

    items_list = sorted(items)
    print(len(items_list))
    print(items_list)

import json

with open("train.jsonl", "r", encoding="utf-8") as f:
    first = json.loads(f.readline())

print(first)


counts = Counter()

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        battle = json.loads(line)
        for mon in battle.get("p1_team_details", []):
            item = mon.get("item")
            if item:  # salta None / stringa vuota
                counts[item] += 1

total = sum(counts.values())
print(f"Item distinti: {len(counts)}  |  Occorrenze totali: {total}\n")

for item, n in counts.most_common():
    print(f"{item:<20} {n:>6}  {n / total:6.1%}")

items_list = sorted(counts)  # la lista degli item unici, se ti serve ancora
'''
import json
from collections import Counter


def count_turns(path):
    """Distribuzione della lunghezza della timeline per battaglia."""
    counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            battle = json.loads(line)
            counts[len(battle.get("battle_timeline", []))] += 1

    total = sum(counts.values())
    print(f"{path}: {total} battaglie, {len(counts)} durate distinte")
    for n_turns, n in sorted(counts.items()):
        print(f"  {n_turns:>3} turni  {n:>6}  {n / total:6.1%}")
    return counts


if __name__ == "__main__":
    count_turns("train.jsonl")
    count_turns("test.jsonl")