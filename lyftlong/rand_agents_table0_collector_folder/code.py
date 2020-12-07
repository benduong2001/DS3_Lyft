import pandas as pd
import numpy as np

file_agents = "rand_agents_table0.csv"
at = pd.read_csv(file_agents)

AT_LEN = len(at)

ROW_AMOUNT = 100000

CSV_AMOUNT = AT_LEN // ROW_AMOUNT + 1

for i in range(CSV_AMOUNT):
    at_i = at.iloc[(i*ROW_AMOUNT) : (i*ROW_AMOUNT) + ROW_AMOUNT]
    at_i.to_csv("rand_agents_table0_part_{0}.csv".format(i))
    print("Part {0} of {1} is done".format(i, CSV_AMOUNT))