MAX_LENGTH_VEHICLES = 19.8 # federally mandated maximum length of vehicles
import pandas as pd
import numpy as np

def agents0_prepare_agent_table(agents_table):
    agents_table = agents0d_filter_oversized_track_ids(agents_table)
    agents_table = agents0c_reduce_columns(agents_table)
    agents_table = agents0b_enhance_yaws(agents_table)
    agents_table = agents0a_filter_onroad(agents_table)
    return agents_table
    
def agents0c_reduce_columns(agents_table):
    agents_table_columns = [
        'centroid_x', 'centroid_y', 'yaw', 'kind', 
        'on_road', 'mean_area', 'speed', 'track_id','frame_index', 'scene_index']
    agents_table = agents_table[agents_table_columns]
    return agents_table

def agents0a_filter_onroad(agents_table: pd.DataFrame) -> pd.DataFrame:
    agents_table_onroad = agents_table[agents_table.on_road == 1]
    return agents_table_onroad

# FEATURE: overbig
def agents0d_filter_oversized_track_ids(agents_table: pd.DataFrame) -> pd.DataFrame:
    global MAX_LENGTH_VEHICLES
    # removed big objects
    # any cars in the dataset that has a sidelength greater than this was probably a data-mistake
    
    temp_mean_at = agents_table.groupby(['scene_index','track_id'], as_index = False).mean()
    overbig_track_ids = temp_mean_at[(temp_mean_at.extent_x > MAX_LENGTH_VEHICLES) |
                                     (temp_mean_at.extent_y > MAX_LENGTH_VEHICLES)]
    overbig_track_ids = overbig_track_ids[["scene_index", "track_id"]]
    merged = pd.merge(agents_table,overbig_track_ids, how='outer', on=["scene_index", "track_id"], indicator=True)
    left_anti_merge = merged[merged['_merge'] == 'left_only']#.drop(columns=["_merged"])
    left_anti_merge = left_anti_merge.drop(columns=["_merge"])


    table = left_anti_merge

    table = table[table.mean_area < 1.7 * 19.8]
    return table

def agents0b_enhance_yaws(agents_table: pd.DataFrame, 
                          yaw_enhancer: float = MAX_LENGTH_VEHICLES/(2*np.pi)) -> pd.DataFrame:
    agents_table['yaw'] *= yaw_enhancer
    return agents_table






def agents1a_get_frame_index_by_frame_list_index(scene_table: pd.DataFrame,
                                                 scene_index: int,
                                                 frame_list_index: int) -> int:
    
    base_frame_index = st[st.scene_index == scene_index]["frame_index_interval_start"].values[0]
    
    frame_index = base_frame_index + frame_list_index
    
    return frame_index
#####
def agents1b_get_scene_index_by_scene_list_index(scene_table: pd.DataFrame, 
                                                 scene_list_index: int) -> int:

    assert scene_list_index >= 0 and scene_list_index <= 100 
    
    scene_index = st.scene_index.values[scene_list_index]
        
    return scene_index

def agents2a_get_agents_table_by_scene_frame_index(agents_table: pd.DataFrame,
                                                   scene_index: [int],
                                                   frame_index: [int]) -> pd.DataFrame:
    agents_table_given_scene = agents_table[agents_table.scene_index.isin(scene_index)]
    agents_table_given_scene_given_frame = agents_table_given_scene[agents_table_given_scene.frame_index.isin(frame_index)]
    return agents_table_given_scene_given_frame
