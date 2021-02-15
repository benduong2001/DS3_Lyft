# Cluster Set Operations
import math

def set_intersection(list_of_sets):
    """
    Gets the intersection of multiple sets
    """
    len_list_of_sets = len(list_of_sets)
    if len_list_of_sets == 0:
        # only activates when the original list of sets is literally empty
        return set([])
    if len_list_of_sets == 1:
        # recursion's base case
        return list_of_sets[0]
    middle = math.floor(len_list_of_sets / 2)
    return set_intersection(list_of_sets[:middle]) & set_intersection(list_of_sets[middle:])
def set_union(list_of_sets):
    """
    Gets the union of multiple sets
    """
    len_list_of_sets = len(list_of_sets)
    if len_list_of_sets == 0:
        # only activates when the original list of sets is literally empty
        return set([])
    if len_list_of_sets == 1:
        # recursion's base case
        return list_of_sets[0]
    middle = math.floor(len_list_of_sets / 2)
    return set_union(list_of_sets[:middle]) | set_union(list_of_sets[middle:])

def set_intersection(list_of_sets):
    """
    Gets the intersection of multiple sets
    """
    len_list_of_sets = len(list_of_sets)
    if len_list_of_sets == 0:
        # only activates when the original list of sets is literally empty
        return set([])
    if len_list_of_sets == 1:
        # recursion's base case
        return list_of_sets[0]
    return list_of_sets[0] & set_intersection(list_of_sets[1:])
def set_union(list_of_sets):
    """
    Gets the union of multiple sets
    """
    len_list_of_sets = len(list_of_sets)
    if len_list_of_sets == 0:
        # only activates when the original list of sets is literally empty
        return set([])
    if len_list_of_sets == 1:
        # recursion's base case
        return list_of_sets[0]
    return list_of_sets[0] | set_union(list_of_sets[1:])

def set_uniformity(list_of_sets):
    overlap = set_intersection(list_of_sets)
    union = set_union(list_of_sets)
    if len(union) == 0:
        return 0
    else:
        return (len(overlap)/len(union))
