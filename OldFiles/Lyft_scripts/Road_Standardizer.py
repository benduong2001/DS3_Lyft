# ROAD_STANDARDIZATION
import math
import numpy as np

def RS1_recenter(agent_table, center_row):
    # 
    agent_table["centroid_x"] = agent_table["centroid_x"].values - center_row["centroid_x"].values[0]
    agent_table["centroid_y"] = agent_table["centroid_y"].values - center_row["centroid_y"].values[0]
    return agent_table

def RS2_reyaw2(agent_table, center_row):
    # points center entity's yaw to 0 pi, revolving every object around with it
    # thus changing both their location and yaws
    # Will utilize rotation matrix and matrix multiplication
    YAW_ANCHOR = (0)*np.pi # make zero to match zeropadding
    center_yaw = center_row["yaw"].values[0]
    center_yaw_difference = YAW_ANCHOR - center_yaw
    
    rotation_matrix = np.array([
        [math.cos(center_yaw_difference),-math.sin(center_yaw_difference)],
        [math.sin(center_yaw_difference), math.cos(center_yaw_difference)]
    ])
    
    agent_table_rotation_matrix_input = agent_table[["centroid_x", "centroid_y"]].to_numpy().T
    
    assert agent_table_rotation_matrix_input.shape[0] == 2
    
    agent_table_rotation_matrix_output = rotation_matrix @ agent_table_rotation_matrix_input
    
    # revolve every surrounding agent by center_yaw_difference    
    # reconvert new_thetas and radii for the newly recentered agent table by polar coordinates
    
    agent_table["centroid_x"] = agent_table_rotation_matrix_output[0]
    agent_table["centroid_y"] = agent_table_rotation_matrix_output[1]
    
    agent_table["yaw"] += center_yaw_difference
    agent_table["yaw"] %= 2*math.pi # re-bases them
    
    return agent_table

def road_standardizer(agent_table,
                      cluster_sequences):
    anchor = cluster_sequences.anchor # center track_id
    center_row = agent_table[agent_table.track_id == anchor]
    new_agent_table = RS1_recenter(agent_table, center_row)
    new_agent_table2 = RS2_reyaw2(new_agent_table, center_row)
    return new_agent_table2
