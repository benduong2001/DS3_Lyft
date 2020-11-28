import pandas as pd
import numpy as np

agents_df = pd.read_csv('rand_agents_table0.csv')
frame_df = pd.read_csv('rand_frames_table1.csv')
scene_df = pd.read_csv('rand_scenes_table1.csv')
agents_df = agents_df.merge(frame_df[['frame_index', 'scene_index']], on='frame_index')
small_df = agents_df[agents_df['PERCEPTION_LABEL_CAR'] == 1]

def scene_car_agent_table(scene_index):
    atrnn1s = small_df[small_df['scene_index'] == scene_index]

    agent_parallel_input_series = pd.DataFrame()
    apis = agent_parallel_input_series
    apis_frame_indices = pd.unique(atrnn1s['frame_index'])
    apis['frame_index'] = apis_frame_indices


    ###

    PARTITION_SIZE = 10
    track_ids = pd.unique(atrnn1s['track_id'])
    apis_cols = ['centroid_x', 'centroid_y'] # + ['frame_index'] 
    for i in range(len(track_ids)):
        if i%100 == 0:
            print("{0} of {1}".format(i, len(track_ids)))
        ti = track_ids[i]
        atrnn1sti = atrnn1s[atrnn1s.track_id == ti].reset_index()
        # print(i)
        if atrnn1sti.__len__() < PARTITION_SIZE:
            # print(i, " is bad")
            continue
        # print(i, "is good")
        atrnn1sti = atrnn1sti[apis_cols]
        atrnn1sti = atrnn1sti.add_prefix('ti{0}_'.format(str(ti)))

        width = len(atrnn1sti.columns)
        height = len(apis_frame_indices) - len(atrnn1sti)

        Nan_arr = np.empty((height, width))
        Nan_arr[:] = np.NaN
        padder = pd.DataFrame(Nan_arr, 
                              columns=atrnn1sti.columns)
        atrnn1sti = pd.concat([atrnn1sti, padder]).reset_index()
        apis = pd.concat([apis, atrnn1sti], axis=1)

    ### 

    apis = apis.drop(['frame_index', 'index'], axis=1)
    
    ### get only 10 first rows
    
    apis = apis.iloc[0:10]
    
    apis.to_csv("CAR_COLLECTOR{0}.csv".format(str(scene_index)))

N = len(pd.unique(agents_df['scene_index']))
N = 100

errors = []
for i in range(N):
    try:
        scene_index = pd.unique(agents_df['scene_index'])[i]
        print ("SCENE {0} of 100".format(str(i)))    
        scene_car_agent_table(scene_index)
    except:
        errors.append(pd.unique(agents_df['scene_index'])[i])
        # fills up the error list with the scenes that encountered errors