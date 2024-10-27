import json
import numpy as np
# To combine all the lists into a single list, we can use list comprehension or itertools.chain
from itertools import chain
# Flattening the list of lists into a single list
path='/workspaces/cma_data/cma_data.json'
with open(path, "r") as f:
    data_json = json.load(f)
    channel_list = data_json["coords"]["channel"]
    
    print(len(channel_list))


# index=[5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47] #0,3

index=[ 10, 14, 26, 34, 40, 43,46,47]
new_list = [list(range(i*5, (i + 1) * 5)) for i in index]
new_list.append([240])
flattened_list = list(chain.from_iterable(new_list))
new_channel_list=[channel_list[i] for i in flattened_list]
print(new_channel_list)
new_index=[]
for i in index:
    new_index=new_index.append(list(range(i,(i+1)*5)))
    print(channel_list[i])
    
    
    
        # index=[5,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,41,42,43,44,45,46,47] #0,3 #37,39
        # new_list = [list(range(i*5, (i + 1) * 5)) for i in index]
        # self.channels_list = list(chain.from_iterable(new_list))