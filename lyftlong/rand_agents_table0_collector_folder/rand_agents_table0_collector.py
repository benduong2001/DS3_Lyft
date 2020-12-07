#!/usr/bin/env python
# coding: utf-8

# For collecting the csv files in lyftlong\rand_agents_table0_collector_folder into the final csv table "rand_agents_table0.csv"

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


rand_agents_table0_collector_folder_path = "\\lyftlong\\rand_agents_table0_collector_folder\\"

files = []

for file in os.listdir(rand_agents_table0_collector_folder_path):
    # assert that they are combined in order:
    if "rand_agents_table0_part" not in file:
        continue
    print(file)
    filename_separated = file.split("_")
    number_str = filename_separated[-1]
    
    if "." in number_str: 
        number = int(number_str.split('.')[0])
    else:
        number = int(number_str)
    files.append({"Number": number, "File": file})


# In[ ]:


files_sorting = sorted(files, key = lambda i: i['Number'], reverse=False) 
# sorts the files by number just in case they aren't already in the folder 
files_sorted = [file['File'] for file in files_sorting]
# gets sorted file names


# In[ ]:


full_table = pd.read_csv(files_sorted[0])

for file in files_sorted[1:]:
    filename = rand_agents_table0_collector_folder_path + file
    next_table = pd.read_csv(filename)
    full_table.append(next_table, ignore_index=True)


# In[ ]:


full_table = full_table.drop("Unnamed: 0", axis=1)


# In[ ]:


rand_agents_table0_path = "\\lyftlong\\rand_agents_table0.csv"
full_table.to_csv(rand_agents_table0_path)

