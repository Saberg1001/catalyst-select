import csv
import pathlib

file = pathlib.Path(r'/mnt/d/1-com-chem/gzy-ecust/work/W-V-O/mace/relaxed/result.csv')
with open(file, 'r') as f:
    reader = csv.reader(f)
for row in reader:
    if len(row) >= 3:  # 确保该行至少有3个元素
        print(row[0], row[2])
