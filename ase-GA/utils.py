import pandas as pd
from rich import print

result = pd.read_csv("../result.csv")
print(result.iloc[:, [0, 2]])
