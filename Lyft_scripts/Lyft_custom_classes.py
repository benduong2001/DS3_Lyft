from PreClustering_Agent_Table_Preparations import agents2a_get_agents_table_by_scene_frame_index
from Cluster_Set_Operations import * # set_union, set_intersection, set_uniformity
from Road_Standardizer import * # road_standardizer
import pandas as pd
import numpy as np
import math
from shapely.geometry import Point, Polygon
from Lyft_dataset_imports import agents_table, frames_table, scenes_table, crosswalk_polygon_list


ROTATION_DECISION = ["NONE", "RECENTER_ROT", "RECENTER_NOROT"][2]
ROAD_JUNCTION_DECISION = ["EXCLUDE", "INCLUDE", "INDIFFERENT"][0]

NUMERICAL_FEATURES = ["centroid_x", "centroid_y", "yaw", "speed", "mean_area"][:3] # <- Remove as many as you want
CATEGORICAL_FEATURES = ["track_id", "frame_index"]


PIVOTED_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
RANK = 3
MAX_TRACK_IDS = 20

class Time_Series_Partition:
    "Time series partition"
    def __init__(self,
                 cluster_sequence,
                 input_array: np.ndarray,
                 output_row: np.ndarray):

        self.all_track_ids = cluster_sequence.all_track_ids
        self.anchor = cluster_sequence.anchor
        self.scene_index = cluster_sequence.scene_index
        self.frame_index = cluster_sequence.frame_index
        
        self.cluster_sequence = cluster_sequence
        self.input_array = input_array
        self.output_row = output_row

class Cluster:
    """
    Class object to represent 1 cluster of entities in 1 frame in 1 scene
    """
    def __init__(self, 
                 scene_index: int,
                 frame_index: int,
                 anchor: int,
                 cluster: set([int]),
                ):
        self.scene_index = scene_index
        self.frame_index = frame_index
        self.anchor = anchor # dbscan cluster label OR anchor track_id
        self.track_ids_set = cluster # set of track_ids
        # use global frame_index
    def __repr__(self):
        return "{0}: {1}".format(str(self.anchor), str(self.track_ids_set))

class Clusters_Sequence:
    """
    Sequential Container Class object to a cluster's transformation in time;
    it is basically a list of clusters with the same anchor attribute
    """
    def __init__(self, 
                 clusters: [Cluster],
                 track_ids_set: set = None,
                ):
        self.cluster_sequence = clusters
        self.all_track_ids = track_ids_set
        
        self.anchor = clusters[0].anchor
        self.rank = len(clusters)
        self.scene_index = clusters[0].scene_index
        self.frame_index = clusters[0].frame_index
    def __repr__(self):
        return "".join([repr(cluster)+"\n" for cluster in self.cluster_sequence])

class Partition_Creation_Function_Holder:
    """
    Maintennance Class and Function Holder for making the cluster_sequences into Times Series Partition
    This includes pivoting and dimension standardization, as well as road_standardization
    """
    def __init__(self):
        pass

    def _optimized_cluster_sequencing_pivot0_get_table(self, clusters_sequences : [Clusters_Sequence]):
        global agents_table
        # ONLY VERTICALLY EXTRACTS THE FRAME SUBSET, NOT THE HORIZONTAL TRACKIDS YET
        clusters_sequence = clusters_sequences[0]
        
        scene_index = clusters_sequence.scene_index
        frame_index = clusters_sequence.frame_index
        rank = clusters_sequence.rank

        frame_index_range = list(range(frame_index, frame_index + rank))

        reduced_agent_table = agents2a_get_agents_table_by_scene_frame_index(agents_table,
                                                                             [scene_index],
                                                                             frame_index_range)
        return reduced_agent_table

    def _optimized_cluster_sequencing_pivot1_table_reduce_track_ids(self, agents_table,
                                                                    clusters_sequence: Clusters_Sequence):
        # REDUCES THE 3 FRAMES AGENT TABLE HORIZONTALLY BY CLUSTER
        all_track_ids = clusters_sequence.all_track_ids
        
        
        reduced_agent_table_ti = agents_table[agents_table.track_id.isin(all_track_ids)]
        # helps to reconfigure out-of-radius track ids
        return reduced_agent_table_ti

    def _optimized_cluster_sequencing_pivot2_table_reduce_columns(self, agents_table,
                                                                  features: [str]):        
        reduced_agent_table = agents_table[features]
        return reduced_agent_table

    def _pivot(self, agent_table):
    
        pivot_agent_table = agent_table.pivot(index='frame_index', columns='track_id')
        return pivot_agent_table

    def _optimized_cluster_sequencing_pivot(self,
                                            reduced_agent_table,
                                            clusters_sequences : [Clusters_Sequence]):
        """RETURNS THE PRE-PIVOTED TABLE"""
        global PIVOTED_FEATURES
        
        new_reduced_agent_table = self._optimized_cluster_sequencing_pivot1_table_reduce_track_ids(reduced_agent_table, clusters_sequences)
        

        new_reduced_agent_table = self._optimized_cluster_sequencing_pivot2_table_reduce_columns(new_reduced_agent_table, PIVOTED_FEATURES)

        pre_pivoted_agent_table = new_reduced_agent_table # pivoted_agent_table = self._optimized_cluster_sequencing_pivot3_pivot_table(reduced_agent_table)
        
        return pre_pivoted_agent_table


    
    def _standardizations1a_padding(self, pivot_table, all_track_ids_amount):

        place_holder = 0
        difference = MAX_TRACK_IDS - all_track_ids_amount
        concat = []
        for feature in NUMERICAL_FEATURES:
            pivot_table_col = pivot_table[[feature]]
            pivot_table_col_arr = pivot_table_col.to_numpy()
            dummy_array = np.full((RANK, difference), place_holder)
            padded_multiindex = np.column_stack((pivot_table_col_arr, dummy_array))
            concat.append(padded_multiindex)
            
        padded_pivot_table_arr = np.concatenate((concat),axis=1)
        padded_pivot_table_arr = np.nan_to_num(padded_pivot_table_arr)
        return padded_pivot_table_arr

    def _standardizations1b_capping(self, pivot_table, all_track_ids_amount):

        place_holder = 0
        concat = []
        for feature in NUMERICAL_FEATURES:
            pivot_table_col = pivot_table[[feature]]
            pivot_table_col_arr = pivot_table_col.to_numpy()
            capped_multiindex = pivot_table_col_arr[:,:20]
            concat.append(capped_multiindex)
            
        padded_pivot_table_arr = np.concatenate((concat),axis=1)
        padded_pivot_table_arr = np.nan_to_num(padded_pivot_table_arr)
        return padded_pivot_table_arr

    def _standardizations1_dimensions(self, pivot_table, all_track_ids_amount):
        ### ceilinged standardization / zeropadding only doable AFTER numpy conversion
        ### because what would we even name the dots.
        if MAX_TRACK_IDS > all_track_ids_amount:
            pivot_table_arr = self._standardizations1a_padding(pivot_table, all_track_ids_amount)
        elif MAX_TRACK_IDS < all_track_ids_amount:
            pivot_table_arr = self._standardizations1b_capping(pivot_table, all_track_ids_amount)
        else:
            pivot_table_arr = pivot_table.to_numpy()
            pivot_table_arr = np.nan_to_num(pivot_table_arr)
        return pivot_table_arr

    def _standardizations(self, pivot_table, all_track_ids_amount):
        ### ceilinged standardization / zeropadding only doable AFTER numpy conversion
        ### because what would we even name the dots.
        pivot_table_arr = self._standardizations1_dimensions(pivot_table, all_track_ids_amount)
        return pivot_table_arr

    
    def cluster_sequences_to_time_series_partition(self,
                                             list_clusters_sequences : [Clusters_Sequence]) -> [Time_Series_Partition]:
        global ROTATION_DECISION
        # from vertical iteration to horizontal iteration
        # MAKE SURE THAT clusters_sequences has the same frames ranges
        # getting reducers


        # problem: separated features make it harder to zeropad sequenced endings:
        # 
        agent_table = self._optimized_cluster_sequencing_pivot0_get_table(list_clusters_sequences)

        list_partitions = []
        for clusters_sequence in list_clusters_sequences:
            
            prepivoted_agent_table = self._optimized_cluster_sequencing_pivot(agent_table.copy(), clusters_sequence)

            prepivoted_agent_table.fillna(0) # precautionary before road rotate

##            counts = prepivoted_agent_table.groupby(['frame_index'])['frame_index'].size().values.astype(int)
##            # recall at this stage that frame_index groupby count is still == track_id amount in a given frame
##            print(prepivoted_agent_table.sort_values("frame_index"))
##            assert len(np.unique(counts)) == 1
##            assert False

            # DECISION POINT
            if ROTATION_DECISION == "RECENTER_ROT":
                rotated_agent_table = road_standardizer(prepivoted_agent_table, clusters_sequence)
            if ROTATION_DECISION == "RECENTER_NOROT":
                rotated_agent_table = road_standardizer_no_rotation(prepivoted_agent_table, clusters_sequence)            
            else:
                rotated_agent_table = prepivoted_agent_table

            # DO rotating BEFORE PIVOTING, due to column revamping

            # change anchor track id (without affecting partition object creation) to -1, so it comes first in pivot)
            rotated_agent_table["track_id"] = [(-1 if x == clusters_sequence.anchor else x) for x in rotated_agent_table["track_id"].values] # exa3
            
            pivot_table = self._pivot(rotated_agent_table)
            all_track_ids_amount = len(clusters_sequence.all_track_ids)
            assert all([(len(pivot_table[x].columns) == all_track_ids_amount)
                        for x in ["centroid_x", "centroid_y"]])

            pivot_table_arr = self._standardizations(pivot_table, all_track_ids_amount)

            input_array = pivot_table_arr[:-1]
            output_vect = pivot_table_arr[-1]

            Time_Series_Partition_Object = Time_Series_Partition(clusters_sequence,
                                                                 input_array,
                                                                 output_vect)
            list_partitions.append(Time_Series_Partition_Object)
        return list_partitions

    

            

            

            

            

        

        

class Frame_Clusters:
    """
    Container Class object that holds a list of all clusters with the same frame_index attribute;
    i.e. list of all the Clusters in one given frame
    """
    def __init__(self, 
                 scene_index: int,
                 frame_index: int,
                 frame_clusters: [Cluster]
                ):
        self.scene_index = scene_index
        self.frame_index = frame_index 
        self.frame_clusters = frame_clusters
        
        # assert all(cluster.frame_index == self.frame_index for cluster in self.frame_clusters)
        self.clusters_dict = dict()
        for cluster in frame_clusters:
            self.clusters_dict[cluster.anchor] = cluster
    def __getitem__(self, key):
        return self.clusters_dict[key]
    def __repr__(self):
        return "".join([repr(cluster)+"\n" for cluster in self.frame_clusters])






class Clusters_Sequencing_Creation_Function_Holder:
    """
    Function holder for ways to group cluster sequentially
    """
    def __init__(self, rank: int):
        self.rank = rank

    def _optimized_clustering_reduction(self,
                                 anchor: int,
                                 list_frame_clusters: [Frame_Clusters] = None) -> Clusters_Sequence:
        """Creates Clusters_Sequence quickly; returns None if standards not met"""
        
        intersection_base = list_frame_clusters[0][anchor].track_ids_set
        anchor_track_ids_set_intersection = intersection_base 
        anchor_track_ids_set_union = set()

        anchor_unified_clusters_list = [] # needed for future clusters_sequence object creation
        
        for i in range(len(list_frame_clusters)): # o(n)
            frame_cluster = list_frame_clusters[i]
            if (anchor in frame_cluster.clusters_dict): # o(1)
                
                anchor_cluster = frame_cluster[anchor] # shortcut for initial attributes in future clusters_sequence object creation
                anchor_unified_clusters_list.append(anchor_cluster)

                
                anchor_track_id_set = anchor_cluster.track_ids_set

                anchor_track_ids_set_intersection = anchor_track_ids_set_intersection & anchor_track_id_set
                anchor_track_ids_set_union = anchor_track_ids_set_union | anchor_track_id_set
            else:
                # TEST 1: check if present consecutively
                return None

        if len(anchor_track_ids_set_union) == 0:
            return None

        set_transformation_ratio = len(anchor_track_ids_set_intersection)/len(anchor_track_ids_set_union)
        if not set_transformation_ratio > 0.6:
            # TEST 2: check if present consecutively
            return None

        # included to avoid reuniting anchor_track_ids_sets again later on
        Clusters_Sequence_Object = Clusters_Sequence(anchor_unified_clusters_list, # we would have needed a list comprehension for this
                                                     track_ids_set = anchor_track_ids_set_union)
            
        return Clusters_Sequence_Object





    def _comprehensive_clustering_reduction1_consecutive_presence(self, anchor, list_frame_clusters) -> bool:
        """verifies that the anchor is in all frame_clusters, i.e. consecutive presence"""
        return all([(anchor in frame_cluster.clusters_dict) for frame_cluster in list_frame_clusters])
        
    def _comprehensive_clustering_reduction2_set_uniformity(self, anchor, list_frame_clusters):
        """verifies that the track_ids sets have a set uniformity of above 0.5"""
        all_anchor_track_ids_sets = [frame_cluster[anchor].track_ids_set for frame_cluster in list_frame_clusters]
        return set_uniformity(all_anchor_track_ids_sets) > 0.6

    def _comprehensive_clustering_reduction3_clusters_sequence_creation(self, anchor, list_frame_clusters):
        """verifies that the track_ids sets have a set uniformity of above 0.5"""
        anchor_clusters = [frame_cluster[anchor] for frame_cluster in list_frame_clusters]
        all_anchor_track_ids_sets = [frame_cluster[anchor].track_ids_set for frame_cluster in list_frame_clusters]
        
        all_anchor_track_ids_sets_union = set_union(all_anchor_track_ids_sets)
        
        Clusters_Sequence_Object = Clusters_Sequence(anchor_clusters, all_anchor_track_ids_sets_union)
        return Clusters_Sequence_Object
        
    def _comprehensive_clustering_reduction(self,
                                 anchor: int,
                                 list_frame_clusters: [Frame_Clusters] = None) -> Clusters_Sequence:
        """Creates Clusters_Sequence comprehensively (i.e. easier to fix) but slower; returns None if standards not met"""
        if not self._comprehensive_clustering_reduction1_consecutive_presence(anchor, list_frame_clusters):
            return None

        if not self._comprehensive_clustering_reduction2_set_uniformity(anchor, list_frame_clusters):
            return None

        # all standards met

        Clusters_Sequence_Object = self._comprehensive_clustering_reduction3_clusters_sequence_creation(anchor, list_frame_clusters)

        return Clusters_Sequence_Object
    

    def frame_clusters_to_clusters_sequence(self,
                                                 list_frame_clusters: [Frame_Clusters]) -> [Clusters_Sequence]:
        """main body for making Clusters_Sequence for k = 3"""
        # vertical iteration
        # list_frame_clusters is always 3,or some small k amount

        first_frame_clusters = list_frame_clusters[0]
        anchors = first_frame_clusters.clusters_dict.keys()
        list_clusters_sequences = []
        # print("fc", [len(x.frame_clusters) for x in list_frame_clusters])
        
        for anchor in anchors:
            try:
                # assert False
                Clusters_Sequence_Object = self._optimized_clustering_reduction(anchor, list_frame_clusters)
                if Clusters_Sequence_Object != None:
                    #  print("cs", sorted(Clusters_Sequence_Object.all_track_ids))
                    list_clusters_sequences.append(Clusters_Sequence_Object)
                else:
                    # print("cs", "None")
                    pass
            except:
                # print("broke")
                Clusters_Sequence_Object = self._comprehensive_clustering_reduction(anchor, list_frame_clusters)
                
                if Clusters_Sequence_Object != None:
                    print("cs", sorted(Clusters_Sequence_Object.all_track_ids))
                    list_clusters_sequences.append(Clusters_Sequence_Object)
                else:
                    print("cs", "None")
                    pass
        return list_clusters_sequences # extended with (248 - rank) others * 100 others

    ###############

    


    

# get scene frame agents as table
# if dbscan: collapse 
# if agent anchor: expand + remove if near intersection
# frame cluster list
# if dbscan:
# if agent anchor:
#
# for both:
# for each unique anchor in 1st frame cluster
#       see if anchor present in other anchors
#           if not: drop
#           if yes: check through set deterioration
# if success: reconstruct prepivot table w/ 1st frame_cluster's frame, scene
#     re-merge coordinates, yaw, speed,  (agent rows)
#     note that this can fix set deterioration, if loose track_ids have moments outside original cluster
#



MAX_RADIUS = 19.8

class Clustering_Algorithm:
    def __init__(self):
        pass
    def table_to_frame_clusters(self, table):
        pass

class Agent_Anchored_Clustering_Algorithm (Clustering_Algorithm):
    def __init__(self):
        pass
    def _optimized_agent_anchored1_radius_clustering(self, anchor_track_id: int,
                                          anchor_agents_table: pd.DataFrame,
                                          max_radius: int = MAX_RADIUS) -> Cluster:
        global crosswalk_polygon_list
        # assert that yaw, centroid_x, centroid_y are in the anchor_agents_table
        anchor_row = anchor_agents_table[anchor_agents_table.track_id == anchor_track_id]
        anchor_scene_index  = anchor_row["scene_index"].values[0]
        anchor_frame_index  = anchor_row["frame_index"].values[0]    
        anchor_x   = anchor_row["centroid_x"].values[0]
        anchor_y   = anchor_row["centroid_y"].values[0]
        anchor_yaw = anchor_row["yaw"       ].values[0]


        
        # INTERSECTIONLESS
        # point = Point(anchor_x, anchor_y)
        # if not all([(poly.distance(point) < max_radius) for poly in crosswalk_polygon_list]):
        #    return None
        
        # get differences of every other agent in the frame (anchor_agents_table) in terms of values above
        
        x_diff   = (anchor_agents_table["centroid_x"].values - anchor_x)**2
        y_diff   = (anchor_agents_table["centroid_y"].values - anchor_y)**2
        yaw_diff = (anchor_agents_table["yaw"].values - anchor_yaw)**2
        
        anchor_dist = np.sqrt(x_diff + y_diff + yaw_diff)

        anchor_agents_table_temp = anchor_agents_table.copy()
        anchor_agents_table_temp["anchor_dist"] = anchor_dist
        
        anchor_agents_table_temp = anchor_agents_table_temp[anchor_agents_table_temp.anchor_dist < max_radius]
        
        cluster = Cluster(
                          scene_index = anchor_scene_index,
                          frame_index = anchor_frame_index,
                          anchor = anchor_track_id,
                          cluster = set(anchor_agents_table_temp["track_id"].values))
        return cluster

    
    def _comprehensive_agent_anchored2_intersection_proximity(self, anchor_track_id, anchor_agents_table, max_radius = MAX_RADIUS):
        # global crosswalk_polygon_list
        anchor_row = anchor_agents_table[anchor_agents_table.track_id == anchor_track_id]   
        # anchor_x   = anchor_row["centroid_x"].values[0]
        # anchor_y   = anchor_row["centroid_y"].values[0]
        # point = Point(anchor_x, anchor_y)
        # near_crosswalks = any([(poly.distance(point) < max_radius) for poly in crosswalk_polygon_list])

        near_crosswalks = bool(anchor_row["near_crosswalks"].values[0])
        return near_crosswalks

    def _comprehensive_agent_anchored(self,
                                          anchor_track_id: int,
                                          anchor_agents_table: pd.DataFrame):
        global ROAD_JUNCTION_DECISION

        
        cluster = self._optimized_agent_anchored1_radius_clustering(anchor_track_id, anchor_agents_table)
        # DECISION POINT
        if ROAD_JUNCTION_DECISION == "INDIFFERENT":
            return cluster # doesn't matter if have crosswalk
        
        near_crosswalk_boolean = self._comprehensive_agent_anchored2_intersection_proximity(anchor_track_id, anchor_agents_table)
        # DECISION POINT
        if ROAD_JUNCTION_DECISION == "EXCLUDE":
            # remove any crosswalks moments
            if near_crosswalk_boolean:
                return None
            else:
                return cluster
        else:
            # wants crosswalks ONLY
            if near_crosswalk_boolean:
                return cluster
            else:
                return None            

    def table_to_frame_clusters(self, selected_scene_frame_agents_table):
        
        anchors = selected_scene_frame_agents_table.track_id.values
        # anchors = selected_scene_frame_agents_table[selected_scene_frame_agents_table.mean_area <= 1.7 * 8].track_id.values
        clusters = []
        for anchor_track_id in anchors:
            cluster = self._comprehensive_agent_anchored(anchor_track_id, selected_scene_frame_agents_table)
            if cluster == None:
                continue
            else:
                clusters.append(cluster)

        # print(clusters)
        if clusters == []: # entirely in junction
            return None
        scene_index = clusters[0].scene_index
        frame_index = clusters[0].frame_index
        Frame_Clusters_Object = Frame_Clusters(scene_index,
                                               frame_index,
                                               clusters)
        return Frame_Clusters_Object

