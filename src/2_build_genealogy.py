
import json, os, subprocess, re, csv, sys, gc, whatthepatch
import pickle
from collections import OrderedDict
import pydriller, time
import networkx as nx
import pandas as pd
from tqdm import tqdm

sys.path.append("..")

# Run shell command from a string
from pydriller import Git


"""
    Functionality: Maps clone boundaries from an old version of code to a new version based on line mappings.
                    Calculates the new start and end lines of a clone in the new version of the code,
                    considering line mappings provided as input. Additionally, calculates churn,
                    representing the number of lines changed within the clone boundaries.

    Expected Input:
    clone_start_old (int): The start line of the clone in the old version of the code.
    clone_end_old (int): The end line of the clone in the old version of the code.
    line_mapping (list): A list of tuples representing line mappings between the old and new versions of the code.

    Expected Output:
    clone_start_new (int): The start line of the clone in the new version of the code.
    clone_end_new (int): The end line of the clone in the new version of the code.
    churn (int): The number of lines changed within the clone boundaries.
"""
# map line number from old commit to new commit
# '47bec44' '672b27d'
def get_mapped_clone(clone_start_old, clone_end_old, line_mapping):
    clone_start_new = clone_start_old
    clone_end_new = clone_end_old

    churn = 0
    begin_to_count = False

    for line_pair in line_mapping:
        old_line = line_pair[0]
        new_line = line_pair[1]

        if old_line is None:
            old_line = 0

        if old_line > clone_end_old:
            return clone_start_new, clone_end_new, churn

        if old_line and new_line:
            # find the new start line
            if old_line <= clone_start_old:
                clone_start_new = (new_line - old_line) + clone_start_old
                clone_end_new = (new_line - old_line) + clone_end_old

                # calculate the new end line
            elif old_line <= clone_end_old:
                begin_to_count = True
                clone_end_new = (new_line - old_line) + clone_end_old
        else:
            if old_line >= clone_start_old:
                begin_to_count = True

            # if last line deleted in the clone boundary
            if begin_to_count:
                if new_line is None:
                    clone_end_new -= 1
                churn += 1
    return clone_start_new, clone_end_new, churn


"""
    Functionality: Maps clone groups to their corresponding modified files in a commit.
                    Adjusts clone boundaries based on line mappings between old and new versions of files.
                    Calculates churn for each modified clone group and total churn for all groups.

    Expected Input:
    clone_group_tuple (tuple): A tuple containing clone group information, where each element represents a clone.
    commit_modified_files (list): A list of tuples representing modified files in the commit.
                                   Each tuple contains the old and new paths of the file and the diff content.

    Expected Output:
    group_modified_list (tuple): A tuple containing modified clone group information after adjusting clone boundaries.
                                  Each element represents a modified clone group.
    churn_all (int): The total churn across all modified clone groups.
"""
def get_mapped_group(clone_group_tuple, commit_modified_files):
    group_modified_list = list()
    churn_all = 0
    for clone in clone_group_tuple:
        clone_path = clone.split(":")[0]
        clone_range = clone.split(":")[1]
        clone_start = int(clone_range.split('-')[0])
        clone_end = int(clone_range.split('-')[1])
        churn = 0
        for modified_file in commit_modified_files:
                # check if the changed file is related to the clones in the clone group
                if clone_path == modified_file[0]:  # old path有调整
                    # 获取new_path
                    if modified_file[1] is None: # new path == None, file being deleted, 当前clone已经不存在
                        clone_end = -1
                        churn = (clone_end - clone_start) + 1
                    else: # simply update
                        clone_path = modified_file[1]  # new path
    
                        for diff1 in whatthepatch.parse_patch(modified_file[2]):  # only one element in the generator
                            line_mapping = diff1[1]
                            clone_start, clone_end, churn = get_mapped_clone(clone_start, clone_end, line_mapping)
                            #line_mapping = next(whatthepatch.parse_patch(modified_file[2]))[1] # only one element in the generator
                            #clone_start, clone_end, churn = get_mapped_clone(clone_start, clone_end, line_mapping)
    
                    break  # check next clone
    
        # clone_start, clone_end 
        if clone_start <= clone_end:
            group_modified_list.append("%s:%d-%d" % (clone_path, clone_start, clone_end))
            churn_all += churn
    return tuple(group_modified_list), churn_all


"""
    Functionality: Retrieves the genealogy of a clone group across a series of commits.
                    Adjusts clone group boundaries and calculates churn for each commit.
                    Matches clone groups with their corresponding clone classes in each commit.

    Expected Input:
    clone_group_tuple (tuple): A tuple containing clone group information, where each element represents a clone.
    commit_list_sliced (list): A list of commit IDs to consider for retrieving clone group genealogy.
    clone_class_dict (dict): A dictionary containing clone classes for each commit.
    commit_modification_dict (dict): A dictionary containing modified files for each commit.

    Expected Output:
    clone_group_genealogy_list (list): A list containing the genealogy of the clone group across commits.
                                       Each element represents a commit where the clone group was modified,
                                       along with churn and the corresponding clone class.
"""
# apply dfs searching for the same group
def retrieve_clone_group_genealogy(clone_group_tuple, commit_list_sliced, clone_class_dict, commit_modification_dict):
    clone_group_genealogy_list = list()

    for commit_id in commit_list_sliced:  # consider the start commit_id
        churn_all = 0

        if commit_modification_dict[commit_id]:
            clone_group_tuple, churn_all = get_mapped_group(clone_group_tuple, commit_modification_dict[commit_id])

        for group in clone_class_dict[commit_id]:
            if set(group).intersection(set(clone_group_tuple)):  # is_clone_group_matched(group, clone_group_tuple):
                clone_class_dict[commit_id].remove(group)
                cnt_siblings = len(set(group))
                clone_group_genealogy_list.append("%s:%d:%d:%s" % (commit_id, churn_all, cnt_siblings, "|".join(group)))
                break  # stop when matched

    return clone_group_genealogy_list


"""
    Functionality: Retrieves modifications for all commits in a sliced commit list.
                    Filters modified files to include only Java files.
                    Stores modification information in a dictionary.

    Expected Input:
    commit_list_sliced (list): A list of commit IDs to retrieve modifications for.
    git_repo: An object representing the Git repository.

    Expected Output:
    commit_modification_dict (dict): A dictionary containing modification information for each commit.
                                     Keys are commit IDs, and values are lists of modified Java files,
                                     each represented as a list containing the old path, new path,
                                     and diff content.
"""
def get_all_commit_modifications(commit_list_sliced, git_repo):
    commit_modification_dict = dict()
    for commit_id in commit_list_sliced:
        
        commit_modification_dict[commit_id] = list()
        commit_modified_files = git_repo.get_commit(commit_id).modified_files
        
        commit_modified_files_java = list(filter(lambda x: x.old_path and x.old_path.endswith('.java'),
                                                 commit_modified_files))
        for modified_java_file in commit_modified_files_java:
            commit_modification_dict[commit_id].append([modified_java_file.old_path, modified_java_file.new_path, modified_java_file.diff])
        
        print("finish processing commit_id: ", commit_id)
    
    return commit_modification_dict


"""
    Functionality: Merges overlapping clone groups into larger connected components.

    Expected Input:
    clone_groups (list): A list of lists, where each inner list represents a clone group.

    Expected Output:
    merged_groups (list): A list of merged clone groups, where each inner list represents a connected component.
"""
# some clone groups detected are acturally the same clone group
def merge_groups(clone_groups):
    #L = [['a', 'b', 'c'], ['b', 'd', 'e'], ['k'], ['o', 'p'], ['e', 'f'], ['p', 'a'], ['d', 'g']]

    graph = nx.Graph()
    # Add nodes to Graph
    graph.add_nodes_from(sum(clone_groups, []))
    # Create edges from list of nodes
    q = [[(group[i], group[i + 1]) for i in range(len(group) - 1)] for group in clone_groups]

    for i in q:
        # Add edges to Graph
        graph.add_edges_from(i)

    # Find all connnected components in graph and list nodes for each component
    return [list(i) for i in nx.connected_components(graph)]


"""
    Functionality: Filters out test functions from clone groups, merges overlapping clone groups into larger connected components,
                   and tuplizes the clone groups at each commit.

    Expected Input:
    clone_class_dict (dict): A dictionary containing clone classes for each commit.
    
    Expected Output:
    clone_class_dict_tuplized (dict): A dictionary containing tuplized and merged clone groups for each commit.
"""
def filter_merge_tuplize(clone_class_dict):
    clone_class_dict_tuplized = dict()

    for commit_id in clone_class_dict:
        commit_groups = list()

        #filter out test functions
        for clone_group in clone_class_dict[commit_id]:
            clone_group_list = list()
            for clone in clone_group:
                clone[0] = os.path.normpath(clone[0])
                if clone[0].find('test') == -1:
                    clone_str = ':'.join(clone)
                    clone_group_list.append(clone_str)

            if clone_group_list:
                commit_groups.append(clone_group_list)

        commit_groups_merged = merge_groups(commit_groups)

        # tuplize the clone groups at a certain commit
        commit_groups_merged_tuplized = list()
        for clone_group in commit_groups_merged:
            commit_groups_merged_tuplized.append(tuple(clone_group))
        clone_class_dict_tuplized[commit_id] = commit_groups_merged_tuplized

    return clone_class_dict_tuplized


"""
    Functionality: Tuplizes clone groups in a dictionary containing clone classes for each commit.

    Expected Input:
    clone_class_dict (dict): A dictionary containing clone classes for each commit.
    
    Expected Output:
    clone_class_dict_tuplized (dict): A dictionary containing tuplized clone groups for each commit.
"""
def tuplize_dict(clone_class_dict):
    clone_class_dict_tuplized = dict()
    for commit_id in clone_class_dict:
        clone_class_dict_tuplized[commit_id] = list()
        for clone_group in clone_class_dict[commit_id]:
            clone_group_list = list()
            for clone in clone_group:
                clone[0] = os.path.normpath(clone[0])
                clone_str = ':'.join(clone)
                clone_group_list.append(clone_str)

            if clone_group_list:
                clone_class_dict_tuplized[commit_id].append(tuple(clone_group_list))
    return clone_class_dict_tuplized
  

"""
Main function to execute the program.
"""
if __name__ == '__main__':
    if len(sys.argv) > 1:
        project = sys.argv[1]
    
    # specify the path for the genealogy
    with open('../data/clones/%s_genealogies_tmp.csv' % project, 'w') as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(['clone_group_tuple', 'start_commit', 'genealogy'])

        # directory paths
        # load clone classes
        with open('../data/clones/%s_clone_result_purified_with_paratype.json' % project, 'r') as clone_jsonfile:
            clone_class_dict = filter_merge_tuplize(json.load(clone_jsonfile, object_pairs_hook=OrderedDict))
            clone_class_dict_nonzero = dict((k, v) for k, v in clone_class_dict.items() if v) # filter out commits with empty clone groups
            commit_list = list(clone_class_dict_nonzero)[::-1]
            commit_list_sliced = list(clone_class_dict_nonzero)[::-1] # initially all commits base on time ascending

            commit_modification_dict = dict()
            git_repo = Git('../data/subject_systems/%s' % project)
            commit_modification_dict = get_all_commit_modifications(commit_list_sliced, git_repo)
            print("building genealogy...")
            for commit_id in tqdm(commit_list):
                # dfs
                # slice from the start_commit

                commit_list_sliced.remove(commit_id) # start commit will not be taken into account
                
                for clone_group_tuple in clone_class_dict_nonzero[commit_id]:
                     # dfs
                     genealogy_list = retrieve_clone_group_genealogy(clone_group_tuple,
                                                                     commit_list_sliced,
                                                                     clone_class_dict_nonzero,
                                                                     commit_modification_dict)
                     # mark as analysed
                     # clone_group_genealogy_set.add(clone_group_tuple)
                     #breakpoint()

                     if genealogy_list:
                         #print(genealogy_list)
                         output_writer.writerow(["|".join(clone_group_tuple), commit_id, ';'.join(genealogy_list)])
                         # commit1, commit2, commit3

                # after this commit_id is traversed, remove it from the commit list which need to loop through
                # commit_list_sliced.remove(commit_id)
            print("finish building genealogy...")

    genealogy_df = pd.read_csv('../data/clones/%s_genealogies_tmp.csv' % project)
    genealogy_df_distinct = genealogy_df.groupby('clone_group_tuple', as_index=False).agg(
        {'start_commit': list, 'genealogy': list})
    genealogy_df_distinct['genealogy'] = genealogy_df_distinct['genealogy'].apply(lambda x: ";".join(x))
    genealogy_df_distinct['start_commit'] = genealogy_df_distinct['start_commit'].apply(lambda x: x[0])
    genealogy_df_distinct.to_csv('../data/clones/%s_genealogies.csv' % project, index=False)

