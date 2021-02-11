from PreClustering_Agent_Table_Preparations import * # agents2a_get_agents_table_by_scene_frame_index
from Lyft_custom_classes import *
from Lyft_dataset_imports import * 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import multiprocessing


RANK = 3 # Keep as 3 because rotation matrix of road_standardizer is 3x3

AT = agents_table
FT = frames_table
ST = scenes_table

class Input_Data_Table_Executor:
    # input: agent table, output: list of partitions
    # 100 scenes
    #  -> 248 frame_clusters            ---X
    #     248 - 3 frame_clusters GROUPS ---X
    #     -> N cluster_sequences        ---X
    #        N partitions               --> scenes  
    def __init__(self, scenes_table = ST,
                       frames_table = FT,
                       agents_table = AT):
        self.partitions_dataset = []
        
        self.Partition_Creation_Function_Holder_Object = Partition_Creation_Function_Holder()
        self.Clusters_Sequencing_Creation_Function_Holder_Object = Clusters_Sequencing_Creation_Function_Holder(RANK)
        self.Agent_Anchored_Clustering_Algorithm_Object = Agent_Anchored_Clustering_Algorithm()
        
        self.scenes_table = scenes_table
        self.frames_table = frames_table
        self.agents_table = agents_table
        
        
    def iterate_cluster_sequences_partitions(self, list_frame_clusters):
        list_clusters_sequences = self.Clusters_Sequencing_Creation_Function_Holder_Object.frame_clusters_to_clusters_sequence(list_frame_clusters)
            
        if list_clusters_sequences == []:
            return None
        list_partitions = self.Partition_Creation_Function_Holder_Object.cluster_sequences_to_time_series_partition(list_clusters_sequences)
        if list_partitions == []:
            print("this line should never be printed, if it does, something really bad happened in partitioning")
            return None
        return list_partitions

    def iterate_frames(self, agent_table_scene_selection, frame_indices):


        frame_clusters_of_scene = [] # should be 248 in length
        
        cluster_sequence_partitions = []

        
        for i in range(len(frame_indices)):
            frame_index = frame_indices[i]
            agent_table_scene_frame_selection = agent_table_scene_selection[agent_table_scene_selection.frame_index == frame_index]
            Frame_Clusters_Object = self.Agent_Anchored_Clustering_Algorithm_Object.table_to_frame_clusters(agent_table_scene_frame_selection)
            if Frame_Clusters_Object == None: # entirely intersection
                print("Frame_index {0} is wholy junctions, discard".format(frame_index))
                pass
            else:
                frame_clusters_of_scene.append(Frame_Clusters_Object)
        
        
        for i in range(len(frame_indices) - RANK):
            list_frame_clusters = frame_clusters_of_scene[i: i + RANK]
            assert list_frame_clusters != []
            partitions = self.iterate_cluster_sequences_partitions(list_frame_clusters)
            if partitions == None:
                pass
            else:
                cluster_sequence_partitions.extend(partitions)

        return cluster_sequence_partitions
            
    def iterate_scenes(self,
                       scene_list_indices = None,
                       frame_list_index_start = ((248//8) * 3),
                       frame_list_index_end   = ((248//8) * 5)):
        if scene_list_indices == None:
            scene_list_indices = list(range(100))

        scene_partitions = []

        scene_indices = self.scenes_table.scene_index.values
        
        
        for i in scene_list_indices:
            scene_index = scene_indices[i]
            
            agent_table_scene = self.agents_table[self.agents_table.scene_index == scene_index]
            
            scene_row = self.scenes_table[self.scenes_table.scene_index == scene_index]
            frame_start = scene_row["frame_index_interval_start"].values[0]
            frame_end = scene_row["frame_index_interval_end"].values[0]
            frame_indices = list(range(frame_start, frame_end))[frame_list_index_start : frame_list_index_end]

            partitions = self.iterate_frames(agent_table_scene, frame_indices)
            if partitions == None:
                pass
            else:
                scene_partitions.extend(partitions)

##                
##        f = open("inner_record.txt", "r")
##        text = f.read()
##        text += "\ndone iterating scenes {0}, amount is {1}".format(scene_list_indices[-1],
##                                                                  len(scene_partitions))
##        f = open("inner_record.txt", "w")
##        f.write(text)
##        f.close()
##        
        return scene_partitions
    

RANK = 3 

SCENES_AMOUNT = 100
SESSIONS = 25
workers_per_session = 4
scenes_per_worker = 1
scenes_per_session = workers_per_session * scenes_per_worker #SCENES_AMOUNT//SESSIONS 


def multiproc_serialize_dataset(scenes_list_index: int, scene_partitions_dataset):
    # scenes_list_index_group is from 0 to 10
    print("e")
    scene_partitions = Input_Data_Table_Executor_Object.iterate_scenes(list(range(scenes_list_index,
                                                                                  scenes_list_index + scenes_per_worker)))
    print("\t\t", len(scene_partitions))

    scene_partitions_dataset.extend(scene_partitions)
    print("\t", len(scene_partitions_dataset))
    
pickle_file_name = "lyft_partitions.pkl"
def create_file():
    pickle_file = open(pickle_file_name, 'wb')   # Pickle file is newly created where foo1.py is
    pickle.dump([], pickle_file)          # dump data to f
    pickle_file.close()
create_file()

Input_Data_Table_Executor_Object = Input_Data_Table_Executor()

if __name__ == "__main__":

    manager = multiprocessing.Manager()
    scene_partitions_dataset = manager.list()

    def session(session_int):
        scene_start_index = scenes_per_session * session_int
        scene_end_index = scenes_per_session * (session_int + 1)

        print("uploading scenes {0} to {1}".format(scene_start_index, scene_end_index))


        jobs = []

        args_iter = [i for i in range(scene_start_index, scene_end_index, scenes_per_worker)]
        
        for i in range(workers_per_session):
            p = multiprocessing.Process(target=multiproc_serialize_dataset, args=(args_iter[i], scene_partitions_dataset))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

##        worker_pool = multiprocessing.Pool()
##        args_iter = [i for i in range(scene_start_index, scene_end_index, scenes_per_worker)]
##        worker_pool.map(multiproc_serialize_dataset, args_iter)
        print(len(scene_partitions_dataset))

        
        print("DO NOT HALT, OVERWRITING FILE CURRENTLY")
        f = open(pickle_file_name, 'rb')   # 'r' for reading; can be omitted
        stored_data = pickle.load(f) 
        stored_data = [x for x in scene_partitions_dataset]
        f = open(pickle_file_name, 'wb') 
        pickle.dump(stored_data, f)          # dump data to f
        f.close()
        print("OVERWRITING FILE DONE; YOU MAY NOW HALT")
    
        
    for i in range(SESSIONS):
        
        session(i)
