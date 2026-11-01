import json
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


if __name__ == "__main__":
    #retrieve_items()
    print(first)