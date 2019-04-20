# -*- coding: utf-8 -*-
# 提取speaker
import csv
import codecs
import json
from collections import Counter

counter = Counter()
process_file_name = '2019'

with codecs.open("data/{}_speaker.json".format(process_file_name), 'w', encoding='utf-8') as w:
    with codecs.open("data/{}.csv".format(process_file_name), 'r', encoding='utf-8') as f:
        csv_file = csv.reader(f)
        for row in csv_file:
            if len(row) == 2:
                if ":" in row[1] and '</' not in row[1]:
                    speaker = row[1].split(":")[0].strip()
                    counter.update([speaker])
        out = {}
        for c in counter:
            out[c] = counter.get(c)
        json.dump(out, w, ensure_ascii=False)

print(counter.most_common(5))
