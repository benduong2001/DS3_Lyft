from PreClustering_Agent_Table_Preparations import * # agents2a_get_agents_table_by_scene_frame_index
from shapely.geometry import Point, Polygon

import pandas as pd
import numpy as np

folderpath = "*"
folderpath = "C:\\Users\\Benson\\Desktop"
file = folderpath + "\\lyftlong\\rand_agents_table0_scene_3kind_onroadDiscrete.csv"
at_orig = pd.read_csv(file)
file = folderpath + "\\lyftlong\\rand_frames_table1.csv"
ft_orig = pd.read_csv(file)
file = folderpath +  "\\lyftlong\\rand_scenes_table1.csv"
st_orig = pd.read_csv(file)


X_COORDINATES_ADJUSTER = 0 + 1300 - 15 
Y_COORDINATES_ADJUSTER = 0 + 2700 + 73 

agents_table = agents0_prepare_agent_table(at_orig.copy())
scenes_table = st_orig.copy()
frames_table = ft_orig.copy()


folderpath = "C:\\Users\\Benson\\Desktop\\"
cw_df_orig = pd.read_csv(folderpath + "lyftlong\\crosswalks_table0.csv")
cw_df = cw_df_orig.copy()


##def adjust_cw_coords(cw_df):
##    cw_df["cw_coord_x"] = cw_df["cw_coord_x"] + X_COORDINATES_ADJUSTER
##    cw_df["cw_coord_y"] = cw_df["cw_coord_y"] + Y_COORDINATES_ADJUSTER
##
##    return cw_df
##
##cw_df = adjust_cw_coords(cw_df)
##cw_df.to_csv(folderpath + "lyftlong\\crosswalks_table0.csv")


cw_df["pairs"] = list(np.dstack((cw_df.cw_coord_x.values, cw_df.cw_coord_y.values))[0] )
polygon_vertices_list = cw_df.groupby('cw_id')["pairs"].apply(list)

crosswalk_polygon_list = [Polygon(polygon_vertices) for polygon_vertices in polygon_vertices_list]
