import sys, os, re, subprocess, shutil
import pandas as pd
import os, subprocess, re, json
from pydriller import Git
import json
import math
import csv
from datetime import datetime
import pandas as pd
# relate metrics to clone group
import pickle
import pandas as pd
from collections import Counter
from collections import defaultdict
import re
import pandas as pd
tree = lambda: defaultdict(tree)
import networkx as nx
import logging
import subprocess
import sys,os
import time
import pandas as pd
import shutil
pd.set_option('display.max_columns', None)


"""
    Functionality: Executes a shell command and returns the output and error messages.
    
    Expected Input:
    command_str (str): The shell command to be executed.
    
    Expected Output:
    cmd_out (bytes): The output of the shell command.
    cmd_err (bytes): The error messages (if any) produced by the shell command.
    """
def shellCommand(command_str):
    cmd =subprocess.Popen(command_str.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    return cmd_out


"""
    Functionality: Sorts the commit IDs by date for a given project.

    Expected Input:
    project (str): The name of the project for which commit IDs are to be sorted.

    Expected Output:
    sorted_commits (list): A list of commit IDs sorted by date.
    """
def sortCommitsByDate(project):
    commit_list = list()
    # the commit logs can be retrieved directly through "git log" command.
    with open('../data/commit_logs/%s_commit_log_df.csv' %project, 'r') as f:
        reader = f.read().split('\n')
        for row in reader:
            elems = row.split(',', 3)
            
            if elems and len(elems) > 1:
                commit_id = elems[0]
                commit_date = re.sub(r'[^0-9]', '', elems[2][:-6])
                if commit_date < '201608010000':
                    commit_list.append([commit_id, commit_date])
    df = pd.DataFrame(commit_list, columns=['commit_id', 'commit_date'])
    sorted_commits = list(df.sort_values('commit_date')['commit_id'])
    return sorted_commits



"""
    Functionality: Extracts parameter types from a function signature.

    Expected Input:
    function_parameters (str): The function parameters as a string.

    Expected Output:
    types_str (str): A comma-separated string containing the extracted parameter types.
    """
def extract_para_types(function_parameters):
    function_parameters = function_parameters.strip()
    if not function_parameters:
        return ""

    function_parameters = function_parameters.replace("final", " ")
    function_parameters = re.sub('\s+', ' ', function_parameters.strip())
    parts = []
    bracket_level = 0
    current = []
    # trick to remove special-case of trailing chars
    for c in (function_parameters + ","):
        if c == "," and bracket_level == 0:
            parts.append("".join(current))
            current = []
        else:
            if c == "<":
                bracket_level += 1
            elif c == ">":
                bracket_level -= 1
            current.append(c)
    parts = [part.strip().split(" ")[0] for part in parts]
    return ",".join(parts)


"""
    Functionality: Extracts clone signatures from clone information.

    Expected Input:
    project (str): The name of the project.
    clone_info_str (str): The clone information string.

    Expected Output:
    clone_signature (list or None): A list containing file path, start-end line numbers, and function signature,
                                    or None if no clone classes are found.
    """
def extract_clone_signiture(project, clone_info_str):
    # clone_info = re.findall(r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"', info_str)
    #clone_info = re.findall(r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\(', clone_info_str)

    idx_right_parenthesis = clone_info_str.find(')')
    '''
    clone_info = re.findall(
        r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\(',
        clone_info_str[:idx_left_parenthesis+1], re.S)
    '''
    clone_info = re.findall(
        r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\)',
        clone_info_str[:idx_right_parenthesis + 1], re.S)
    # print(clone_info)
    if len(clone_info):
        file_path = clone_info[0][0].split('/' + project + '/', 1)[-1]
        #file_path = clone_info[0][0].split(project + '/', 1)[-1]
        startline = clone_info[0][1]
        endline = clone_info[0][2]
        pcid = clone_info[0][3]

        # get function name without parameters
        #func_name = clone_info[0][4].strip().split(' ')[-1]

        # get function name and parameters
        func_header = clone_info[0][4].split('(')
        func_name = func_header[0].strip().split(' ')[-1]
        #breakpoint()
        func_paras = extract_para_types(func_header[1].strip())

            #print(func_header)
            #breakpoint()
        '''
        if func_header[1]:
            func_paras = func_header[1].split(',')
            try:
                func_para_str = ','.join(list(map(lambda para: para.split()[-2], func_paras)))
                func_name += '(' + func_para_str + ')'
            except IndexError:
                if func_header[1] == 'Map<String, Map<String, T>> vars, String scriptName, String key, T value':
                    func_name += '(Map<String, Map<String, T>>,String,String,T)'
                print(commit_id, func_header[1])

        else:
            func_name += '()'
        '''

        return [file_path, startline + '-' + endline, func_name + "(" + func_paras + ")"]
    else:
        #breakpoint()
        #print("no clone classes find: ", commit_id)
        return None


"""
    Functionality: Parses clone detection results for a specific commit.

    Expected Input:
    project (str): The name of the project.
    commit_id (str): The ID of the commit.

    Expected Output:
    result_list (list): A list containing clone groups, each represented by a list of clone signatures.
    """
def parse_clone_result(project, commit_id):
    result_list = list()
    
    commit_clone_result_path = os.path.join(r"C:\Users\Cleo\Downloads\clone_detection_result_from_jzlarge_1220", project, '%s.xml' % commit_id)
    print(commit_clone_result_path)
    # breakpoint()
    if os.path.exists(commit_clone_result_path):
        with open(commit_clone_result_path, 'r', encoding="ISO-8859-1") as f:
        #with open(commit_clone_result_path, 'r', encoding='unicode_escape') as f:
            reader = f.read()
            # print(reader)

            # extract a pair of clones
            group_list = re.findall(
                r'<class classid=\"[0-9]+\" nclones=\"[0-9]+\" nlines=\"[0-9]+\" similarity=\"[0-9]+\">(.+?)</class>',
                reader, re.DOTALL)
            for group in group_list:  # 一个class
                # extract clone pair strings
                clone_group = list()
                clone_info = re.findall(r'<source (.+?)</source>', group, re.DOTALL)
                clone_signiture = ""
                for snippet in clone_info:
                    try:
                        clone_signiture = extract_clone_signiture(project, snippet)
                    except:
                        print("error: ", commit_id)
                    if clone_signiture:
                        clone_group.append(clone_signiture)
                if len(clone_group):
                    result_list.append(clone_group)
    else:
        print("file not found")
    return result_list


"""
    Functionality: Computes the size of a clone pair.

    Expected Input:
    clone_pair (str): The clone pair string in the format "file_path^start_line^end_line+file_path^start_line^end_line".

    Expected Output:
    clone_size (int): The size of the larger clone fragment in the pair.
    """
def computeCloneSize(clone_pair):
    size1 = int(clone_pair.split('+')[0].split('^')[2]) - int(clone_pair.split('+')[0].split('^')[1]) + 1
    size2 = int(clone_pair.split('+')[1].split('^')[2]) - int(clone_pair.split('+')[1].split('^')[1]) + 1
    return max(size1, size2)


"""
    Functionality: Extracts raw genealogies for a given project and tool.

    Expected Input:
    project (str): The name of the project.
    tool (str): The name of the tool.

    Expected Output:
    raw_genealogy_dict (dict): A dictionary containing raw genealogies.
    end_commit_set (set): A set containing end commits.
    """
def raw_genealogies(project, tool):
    raw_genealogy_dict = dict() # use dict to store the genealogy
    end_commit_set = set() # remove the duplicates
    with open('../output_data/%s/%s_genealogies.csv' %(tool,project), 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader, None)
        for row in csvreader:
            genealogy = row[3]
            if len(genealogy):
                clone_pair = row[0]
                size = computeCloneSize(clone_pair)
                path1 = clone_pair.split('+')[0].split('^')[0]
                path2 = clone_pair.split('+')[1].split('^')[0]
                start_commit = row[1]
                end_commit = row[2]
                end_commit_set.add(end_commit)
                raw_genealogy_dict[(path1, path2)] = {'size':size, 'start':start_commit, 'end': end_commit, 'gen':genealogy, 'sig':clone_pair}
    return raw_genealogy_dict, end_commit_set


"""
    Functionality: Finds renamed files for each end commit in the set.

    Expected Input:
    end_commit_set (set): A set containing end commits.
    project (str): The name of the project.

    Expected Output:
    rename_dict (dict): A dictionary containing renamed files for each end commit.
    """
def renamedFiles(end_commit_set):
    rename_dict = dict()
    # directory paths (global variables)
    current_dir = os.getcwd()
    repo_dir = '../src_code/%s' %project
    # change to the code repository directory
    os.chdir(repo_dir)
    for commit_id in end_commit_set:
        #print commit_id
        diff_list = list()
        diff_res = shellCommand('git diff %s^ %s --name-status -M' %(commit_id,commit_id))
        for line in diff_res.split('\n'):
            if line.startswith('R099') or line.startswith('R100'):
                old_path = line.split('\t')[1]
                new_path = line.split('\t')[2]
                diff_list.append((old_path, new_path))
        if len(diff_list):
            rename_dict[commit_id] = diff_list
    # go back to the script directory
    os.chdir(current_dir)
    return rename_dict


"""
    Functionality: Loads commit dates from a file and stores them in a dictionary.

    Expected Input:
    file_name (str): The name of the file containing commit dates.

    Expected Output:
    commit_date_dict (dict): A dictionary containing commit dates, where keys are commit IDs and values are dates.
    """
def loadCommitDate(file_name):
    commit_date_dict = dict()
    with open(file_name, 'r') as f:
        reader = f.read().split('\n')
        for line in reader:
            if len(line):
                elems = line.split(',')
                rev = elems[0]
                date = re.sub(r'[^0-9]', '', elems[2][:-6])
                commit_date_dict[rev] = date
    return commit_date_dict


"""
    Functionality: Calculates the difference in days between two dates.

    Expected Input:
    d1_str (str): The first date string in the format '%Y%m%d%H%M%S'.
    d2_str (str): The second date string in the format '%Y%m%d%H%M%S'.

    Expected Output:
    diff_days (float): The difference in days between the two dates.
    """
def dateDiff(d1_str, d2_str):
    d1 = datetime.strptime(d1_str, '%Y%m%d%H%M%S')
    d2 = datetime.strptime(d2_str, '%Y%m%d%H%M%S')
    return (d2 - d1).total_seconds()/3600/24


"""
    Functionality: Categorizes the interval between two dates into different time periods.

    Expected Input:
    last_date (str): The date string representing the last date.
    current_date (str): The date string representing the current date.

    Expected Output:
    category (str): The category of the interval ('YY' for year, 'Y' for month, 'M' for week, 'W' for day).
    """
def categoriseInterval(last_date, current_date):
    interval = dateDiff(last_date, current_date)
    if interval > 365:
        return 'YY'
    elif interval > 30:
        return 'Y'
    elif interval > 7:
        return 'M'
    elif interval > 1:
        return 'W'
    return 'D'


"""
    Functionality: Combines renamed pairs with raw genealogies.

    Expected Input:
    raw_genealogy_dict (dict): A dictionary containing raw genealogies.
    rename_dict (dict): A dictionary containing renamed files for each end commit.

    Expected Output:
    updated genealogy
    """
def combineRenamedPair(raw_genealogy_dict, rename_dict):
    for paths in raw_genealogy_dict:
        end_commit = raw_genealogy_dict[paths]['end']
        if end_commit in rename_dict:
            for renamed_paths in rename_dict[end_commit]:
                if paths == renamed_paths:
                    print('renamed!')
                    break
    return



"""
    Functionality: Loads fault-inducing commits from a file and organizes them into a dictionary.

    Expected Input:
    project (str): The name of the project.

    Expected Output:
    fault_inducing_dict (dict): A dictionary containing fault-inducing commits and their corresponding fixing commits.
    """
def loadFaultInducingCommits(project):
    fault_inducing_dict = dict()
    with open('../output_data/szz/%s/fault_inducing.csv' %project, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader, None)
        for row in csvreader:
            inducing_commit = row[0]
            fixing_commit = row[1]
            if inducing_commit in fault_inducing_dict:
                fault_inducing_dict[inducing_commit].add(fixing_commit)
            else:
                fault_inducing_dict[inducing_commit] = set([fixing_commit])
    return fault_inducing_dict


"""
    Functionality: Generates features for genealogy analysis.

    Expected Input:
    raw_genealogy_dict (dict): A dictionary containing raw genealogies.
    fault_inducing_dict (dict): A dictionary containing fault-inducing commits and their corresponding fixing commits.
    commit_date_dict (dict): A dictionary containing commit dates.

    Expected Output:
    None
    """
def genealogyFeatures(raw_genealogy_dict, fault_inducing_dict, commit_date_dict):
    output_list = list()
    for paths in raw_genealogy_dict:
        genealogy_list = raw_genealogy_dict[paths]['gen'].split('-')
        # compute time interval between changes
        start_commit_date = commit_date_dict[raw_genealogy_dict[paths]['start']]
        end_commit_date = commit_date_dict[raw_genealogy_dict[paths]['end']]
        last_date = start_commit_date
        interval_list = list()
        churn_list = list()
        # check whether a clone modification is faulty        
        fault_proneness_list = list()
        buggy_gen = False
        commit_list = [raw_genealogy_dict[paths]['start']]
        for mod in genealogy_list:
            state = mod.split(',')[0]
            commit_id = mod.split(',')[1]
            churn = mod.split(',')[2]
            commit_list.append(commit_id)
            current_date = commit_date_dict[commit_id]
            interval = categoriseInterval(last_date, current_date)
            last_date = current_date
            if commit_id in fault_inducing_dict:
                buggy = 'Y'
                # check whether a whole genealogy is faulty
                fixing_commits = fault_inducing_dict[commit_id]
                for f_c in fixing_commits:
                    if commit_date_dict[f_c] > end_commit_date:
                        buggy_gen = True
                        break
            else:
                buggy = 'N'
            fault_proneness_list.append('%s_%s' %(state,buggy))
            interval_list.append('%s_%s' %(interval,buggy))
            churn_list.append(churn)
        fault_proneness_str = '^'.join(fault_proneness_list)
        interval_str = '^'.join(interval_list)
        commit_str = '^'.join(commit_list)
        churn_str = '^'.join(churn_list)
        # signature and size
        sig = raw_genealogy_dict[paths]['sig']
        clone_size = raw_genealogy_dict[paths]['size']
        # output results
        output_list.append([sig, clone_size, buggy_gen, fault_proneness_str, interval_str, commit_str, churn_str])
        df = pd.DataFrame(output_list, columns=['signature', 'size', 'buggy_gen', 'state+fault', 'interval', 'commits', 'churn'])
        df.to_csv('../statistics/%s/%s_basic.csv' %(tool,project), index=False)
    return


"""
    Functionality: Builds the Understand project database for a specific commit.

    Expected Input:
    commit_id (str): The ID of the commit.
    metric_columns (list): A list of metric columns to be analyzed.
    project (str): The name of the project.
    clone_files (list): A list of clone files.

    Expected Output:
    None
    """
def build_und_project_db(commit_id, metric_columns, project, clone_files):
    
    # run understand cli to construct the project understand db
    # create db
    und_commit_db = '../data/udb/%s/%s.und' % (project, commit_id)

    if os.path.exists(und_commit_db):
        shutil.rmtree(und_commit_db, ignore_errors=True)


    # create und db
    cmd_create_und_db = ['und', '-db', und_commit_db, 'create', '-languages', 'Java']
    shellCommand(cmd_create_und_db)

    # add all files into db corresponding to the current commit
    if platform.system() == "Linux":
        und_commit_db += '.udb'
    for clone_file in clone_files:
        if platform.system() == "Linux":
            clone_file = clone_file.replace("\\", "/")
        cmd_add_file = ['und', 'add', clone_file, und_commit_db]
        shellCommand(cmd_add_file)

    shellCommand(cmd_add_file)

    # settings and analyze udb to retrieve functions with parameters
    cmd_setting_analyze = ['und', '-db', und_commit_db, 'settings', '-metrics']
    cmd_setting_analyze.extend(metric_columns)
    cmd_setting_analyze.extend(['-MetricShowFunctionParameterTypes', 'on'])
    cmd_setting_analyze.extend(['-ReportDisplayParameters', 'on', 'analyze', 'metrics'])
    
    shellCommand(cmd_setting_analyze)


"""
    Functionality: Retrieves files associated with a specific commit from the genealogy DataFrame.

    Expected Input:
    commit_id (str): The ID of the commit.
    genealogy_df (DataFrame): The DataFrame containing clone group tuples and start commit IDs.

    Expected Output:
    files_by_commit (set): A set of files associated with the commit.
    """
def get_files_by_commit(commit_id, genealogy_df):
    groups_by_commit = genealogy_df.loc[genealogy_df['start_commit'] == commit_id]['clone_group_tuple']

    files_by_commit = set()
    for group in groups_by_commit:
        for clone in group.split("|"):
            clone_path = os.path.normpath(clone.replace("'", "").strip().split(":")[0])
            
            #clone_str = eval(repr(clone.split("-")[0].replace("'", "").strip())).replace('\\\\', '\\')
            #clone_path = clone_path.replace("%s\\" % project, "") # 去掉前面的project 名称
            # clone = os.path.normpath(clone.replace(".java", "")).replace(os.path.sep, ".")
            # clone_path = os.path.normpath(clone.split(":")[0])
            if len(clone):
                files_by_commit.add(clone_path)
    
    return files_by_commit


"""
    Functionality: Retrieves clone classes for a given project.

    Expected Input:
    project (str): The name of the project.

    Expected Output:
    clone_class_dict_4_clone (defaultdict): A nested dictionary containing clone classes.
    """
def get_clone_class(project):
    clone_class_dict_4_clone = defaultdict(defaultdict)
    project_clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH,
                                                      '%s_clone_result_purified_with_paratype.json' % project)
    # load clone classes
    with open(project_clone_result_purified_path, 'r') as clone_jsonfile:
        clone_result_json = json.load(clone_jsonfile)

        for commit_id in clone_result_json:
            # filter out test functions
            for clone_group in clone_result_json[commit_id]:
                for clone in clone_group:
                    dot_java_idx = clone[0].rfind(".java")
                    
                    clone[0] = clone[0][0:dot_java_idx] + clone[0][dot_java_idx:].replace(".java", "")
                    clone[0] = os.path.normpath(clone[0]).replace(os.path.sep, ".")

                    #clone[0] = os.path.normpath(clone[0].split(".java")[0]).replace(os.path.sep, '.')  # remove .java and replace / with .
                    if clone[0].lower().find('test') == -1:  # filter out test methods
                        clone_signiture = ':'.join(clone[:2])  # clone[2] is the function name

                        clone_class_dict_4_clone[commit_id][clone_signiture] = clone[2]
    
    return clone_class_dict_4_clone


"""
    Functionality: Retrieves metrics for a specific commit of a project.

    Expected Input:
    project (str): The name of the project.
    commit_id (str): The identifier of the commit.

    Expected Output:
    commit_metric_dict (dict): A dictionary containing metrics for methods/functions in the commit.
    """
def get_metrics_by_commit(project, commit_id):
    # commit_metric_df = metric_df.loc[metric_df['commit_id'] == commit_id].drop_duplicates()
    commit_metric_path = os.path.join(config_global.UDB_PATH, "%s" % project, '%s.csv' % commit_id)
    if not os.path.exists(commit_metric_path):
        print("commit metrics not exist: ", commit_id)

    commit_metric_df = pd.read_csv(commit_metric_path)
    print("commit_metric_df: ", commit_metric_df.shape)

    # filter out non-methods
    commit_metric_df = commit_metric_df[commit_metric_df['Kind'].str.contains('method', case=False) |
                                        commit_metric_df['Kind'].str.contains('function', case=False) |
                                        commit_metric_df['Kind'].str.contains('procedure', case=False) |
                                        commit_metric_df['Kind'].str.contains('constructor', case=False)
                                        ]

    if 'Kind' in commit_metric_df.columns:
        commit_metric_df.drop(['Kind'], axis=1, inplace=True)
    if 'File' in commit_metric_df.columns:
        commit_metric_df.drop(['File'], axis=1, inplace=True)
    #print('commit_metric_df after: ', commit_metric_df.shape)

    # process the function signiture column
    pattern = "\\(.*?\\)"
    #commit_metric_df['Name'] = commit_metric_df['Name'].str.replace(pattern, '')

    #commit_metric_df['Name'] = commit_metric_df['Name'].str.replace('\.[a-zA-Z0-9_]+\.\.', '.') # remove (Anon_1)
    commit_metric_df['Name'] = commit_metric_df['Name'].str.replace('.\(Anon_[0-9]+\).', '.')  # remove (Anon_1)

    # filter out empty methods
    commit_metric_df = commit_metric_df[commit_metric_df['CountLine'] > 0]

    # filter out duplicates
    commit_metric_df.drop_duplicates(inplace=True)

    # add range to the clone signiture
    #commit_metric_df['Name'] = commit_metric_df['Name'].str.cat(commit_metric_df['CountLine'].astype(str), sep=':')

    # adjust the path to make it consistent with the clone path in genealogy
    # map to the relative path
    # commit_metric_df['clone_signiture'] = commit_metric_df['clone_signiture'].map(
    # lambda path: os.path.relpath(path, start=os.path.normpath(r'C:\Users\Cleo\Dropbox\876\subject_systems\zaproxy')))

    # since all commit are the same
    # commit_metric_df.drop(['commit_id'], axis=1, inplace=True)

    # convert it to dict with key on clone_signiture
    # overriding functions: retrieve the first match
    commit_metric_df = commit_metric_df.drop_duplicates(subset='Name')
    commit_metric_df.set_index(['Name'], inplace=True)
    commit_metric_dict = commit_metric_df.to_dict('index')

    return commit_metric_dict


"""
    Functionality: Searches for a clone in the commit metric dictionary based on the provided clone signature.

    Expected Input:
    commit_metric_dict (dict): A dictionary containing metrics for methods/functions in the commit.
    clone_str (str): The clone signature to search for.

    Expected Output:
    match_metrics (dict or None): Metrics for the matched method/function if found, else None.
    """
def search_clone(commit_metric_dict, clone_str):
    for key in commit_metric_dict:
        idx = key.find('(')
        key_without_class_info = key[:idx].split(".")
        key_without_class_info.pop(-2)

        key_without_class_info_str = '.'.join(key_without_class_info)
        #print(" key no class: ", key_without_class_info_str)
        # some method from understand tool has class info

        if key in clone_str:
            return commit_metric_dict.get(key)

        if key_without_class_info_str in clone_str:
            return commit_metric_dict.get(key)

        #key_without_class_info.pop(-2) # 调整

        key_without_class_info_str = '.'.join(key_without_class_info)

        if key_without_class_info_str in clone_str:
            return commit_metric_dict.get(key)

    return None


"""
    Functionality: Combines Understand metrics with other metrics for clone groups.

    Expected Input:
    und_metric_on_group_df (DataFrame): DataFrame containing Understand metrics for clone groups.
    project (str): The name of the project.

    Expected Output:
    None
    """
def combine_und_other_metrics(und_metric_on_group_df):
    print("\n\ngroup metric und: ", und_metric_on_group_df.shape)
    # loading other metrics file
    other_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_other_metric.csv' % project)
    other_metric_on_group_df = pd.read_csv(os.path.normpath(other_metric_on_group_path))
    print("group metric other: ", other_metric_on_group_df.shape)

    merged_df = pd.merge(und_metric_on_group_df, other_metric_on_group_df, on='clone_group_tuple', how='inner')
    print("merged: ", merged_df.shape, merged_df.columns)
    merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
    merged_df.to_csv(merged_metric_on_group_path, index=False)


"""
    Functionality: Anatomizes genealogy data by adding various derived columns.

    Expected Input:
    genealogy_df (DataFrame): DataFrame containing genealogy data.
    commit_time_dict (dict): Dictionary mapping commit IDs to commit times.
    commit_author_dict (dict): Dictionary mapping commit IDs to commit authors.

    Expected Output:
    genealogy_df (DataFrame): Updated DataFrame with additional derived columns.
    """
def anatomize_genealogy(genealogy_df, commit_time_dict, commit_author_dict):
    genealogy_df['n_siblings_start'] = genealogy_df['clone_group_tuple'].map(lambda x: len(x.split('|')))

    genealogy_df['genealogy_list'] = genealogy_df['genealogy'].map(lambda gen: gen.split('|'))
    genealogy_df['commit_list'] = genealogy_df['genealogy_list'].map(lambda gen: [x.split(":")[0] for x in gen])
    genealogy_df['churn'] = genealogy_df['genealogy_list'].map(lambda gen: sum([int(x.split(":")[1]) for x in gen]))
    genealogy_df.drop(['genealogy'], axis=1, inplace=True)
    genealogy_df['n_genealogy'] = genealogy_df['commit_list'].map(lambda x: len(x))

    # each commit in genealogy: commit_id: #updates: clone_group_sig
    genealogy_df['end_commit'] = genealogy_df['commit_list'].map(lambda x: x[-1])


    # calculate duration in terms of days
    genealogy_df['n_days'] = list(map(lambda x, y: (commit_time_dict[y] - commit_time_dict[x]).days,
                                      genealogy_df['start_commit'], genealogy_df['end_commit']
                                      )
                                  )  # git_repo.get_commit(commit_id).committer.email for commit_id in gen.split('-')
    genealogy_df['start_timestamp'] = genealogy_df['start_commit'].map(lambda x: commit_time_dict[x])  # get timestamp for time-series training
    print("----------------------------------------------------------------")
    # calucate #authors

    genealogy_df['author_list'] = genealogy_df['genealogy_list'].map(
            lambda gen: set([commit_author_dict[commit.split(':', 2)[0]] for commit in gen])
        )

    genealogy_df['n_authors'] = genealogy_df['author_list'].map(lambda x: len(x))
    genealogy_df['cnt_siblings'] = genealogy_df['genealogy_list'].map(lambda gen: max([int(x.split(":")[2]) for x in gen]))
    print(list(genealogy_df['cnt_siblings']))
    # reorder the columns
    #print(genealogy_df.columns)
    print("before drop duplicates: ", genealogy_df.shape)


    # drop duplicates
    genealogy_df['clone_group_tuple'].drop_duplicates(inplace=True)
    print("after drop duplicates: ", genealogy_df.shape)
    print("==================================")
    return genealogy_df


"""
    Functionality: Ranks the genealogy DataFrame either by the length of commit genealogy or by the number of days.

    Expected Input:
    genealogy_df (DataFrame): DataFrame containing genealogy data.
    is_by_genealogy (bool): Indicates whether to rank by the length of commit genealogy (True) or by the number of days (False).
    threshold (float): Threshold value for determining the quantile.

    Expected Output: None (Updates the DataFrame in place).
    """
def rank_by_lifecycle(genealogy_df, is_by_genealogy, threshold=0.5): # if is_by_genealogy, rank by length of commit genealogy, otherwise by number of days
    '''
        genealogy_df['rank_by_n_genealogy'] = genealogy_df['n_genealogy'].map(
                lambda x: 1 if x > genealogy_df.n_genealogy.quantile(0.75) else (
                    0 if x < genealogy_df.n_genealogy.quantile(0.25) else -1))
    '''

    genealogy_df['rank_by_n_genealogy'] = genealogy_df['n_genealogy'].map(
        lambda x: 1 if x > genealogy_df.n_genealogy.quantile(threshold) else 0)

    genealogy_df['rank_by_n_days'] = genealogy_df['n_days'].map(
        lambda x: 1 if x > genealogy_df['n_days'].quantile(threshold) else 0)

    if is_by_genealogy:
        genealogy_df['rank_by_lifecycle'] = genealogy_df['rank_by_n_genealogy']
    else:
        genealogy_df['rank_by_lifecycle'] = genealogy_df['rank_by_n_days']


"""
    Functionality: Ranks the genealogy DataFrame by the prevalence of clones in the starting commit.

    Expected Input:
    genealogy_df (DataFrame): DataFrame containing genealogy data.
    threshold (float): Threshold value for determining the quantile.

    Expected Output: None (Updates the DataFrame in place).
    """
def rank_by_prevalence(genealogy_df, threshold=0.5):
    #genealogy_df['rank_by_prevalence'] = genealogy_df['n_siblings_start'].map(lambda x: 1 if x > genealogy_df.n_siblings_start.quantile(0.5) else 0)

    genealogy_df['rank_by_prevalence'] = genealogy_df['cnt_siblings'].map(
        lambda x: 1 if x > genealogy_df.cnt_siblings.quantile(threshold) else 0)
    print(list(genealogy_df['rank_by_prevalence']))
    print("rank_by_prevalence: ", genealogy_df['rank_by_prevalence'].value_counts())
    print("----------------")
    #genealogy_df['rank_by_n_authors'] = genealogy_df['rank_by_n_authors'].map({1: 'high', 0: 'low'})  # 'volvo':0 , 'bmw':1, 'audi':2} )


"""
    Functionality: Ranks the genealogy DataFrame by bug proneness.

    Expected Input:
    genealogy_df (DataFrame): DataFrame containing genealogy data.
    buggy_commit_list (list): List of buggy commits.
    threshold (float): Threshold value for determining the quantile.

    Expected Output: None (Updates the DataFrame in place).
    """
def rank_by_bugproneness(genealogy_df, buggy_commit_list, threshold=0.5):
    # look into the distribution of n_genealogy
    segments = pd.cut(genealogy_df['n_genealogy'], bins=[0,2,5,100,1000])
    counts = pd.value_counts(segments, sort=True)
    #print(genealogy_df['n_genealogy'].value_counts())
    print(counts)

    buggy_commit_list = list(set(buggy_commit_list))
    print("genealogy columns: ", genealogy_df.columns)

    genealogy_df['buggy_genealogy'] = genealogy_df['commit_list'].map(lambda gen: list(set(gen).intersection(set(buggy_commit_list))))
    genealogy_df['n_buggy_genealogy'] = genealogy_df['buggy_genealogy'].map(lambda gen: len(gen))

    genealogy_df['bug_proneness'] = genealogy_df.apply(lambda row:  (row['n_genealogy'] - row['n_buggy_genealogy']) / (row['n_buggy_genealogy'] if row['n_buggy_genealogy'] != 0 else 1), axis=1)
    genealogy_df['rank_by_bugproneness'] = genealogy_df['bug_proneness'].map(
        lambda x: 1 if x > genealogy_df.bug_proneness.quantile(threshold) else 0)

    # look into the distribution of bug_proneness
    segments = pd.cut(genealogy_df['bug_proneness'], bins=[0, 2, 5, 10, 100, 10000])


"""
    Functionality: Decides the final label for genealogy DataFrame based on provided dimensions.

    Expected Input:
    genealogy_df (DataFrame): DataFrame containing genealogy data.
    is_longevous (bool): Boolean indicating whether to consider longevity.
    is_prevalent (bool): Boolean indicating whether to consider prevalence.
    is_buggy (bool): Boolean indicating whether to consider bug proneness.

    Expected Output: None (Updates the DataFrame in place).
    """
def decide_final_label(genealogy_df, is_longevous, is_prevalent, is_buggy):
    if is_longevous and is_prevalent and is_buggy: # decide the label by all the three dimensions
        genealogy_df['is_reusable'] = genealogy_df.apply(
            lambda row: math.floor((row.rank_by_lifecycle + row.rank_by_prevalence + row.rank_by_bugproneness) / 3),
            axis=1
        ).astype(int)
    elif is_longevous and is_buggy:
        genealogy_df['is_reusable'] = genealogy_df.apply(
            lambda row: math.floor((row.rank_by_lifecycle + row.rank_by_bugproneness) / 2),
            axis=1
        ).astype(int)
    elif is_longevous:
        genealogy_df['is_reusable'] = genealogy_df['rank_by_lifecycle']


"""
    Functionality: Maps numerical labels to categorical labels in the DataFrame column.

    Expected Input:
    col (Series): DataFrame column containing numerical labels.

    Expected Output: None (Updates the DataFrame column in place).
    """
def map_label(col):
    genealogy_df[col] = genealogy_df[col].map({1: 'high', 0: 'low'})


"""
Main function to execute the program.
"""
if __name__ == '__main__':
    
    DEBUG = False
    
    if len(sys.argv) != 3:
            print('Please input [project] & [tool]!')
    else:
        project = sys.argv[1]
        tool = sys.argv[2]
        
        tool_list = ['nicad', 'iclones']
        if (tool in tool_list):
            # get the current directory
            current_dir = os.getcwd()
            # sort commits by date in ascending order
            sorted_commits = sortCommitsByDate(project)
            # total number of commits
            num_commits = len(sorted_commits)
            # initialisation
            if tool == 'nicad':
                # clean previous clone results and make the results' directory
                subprocess.Popen('rm -rf ../subject_systems/%s_functions*' %project, shell=True)
                subprocess.Popen('mkdir -p ../clone_results/nicad/%s' %project, shell=True)
            elif tool == 'iclones':
                subprocess.Popen('mkdir -p ../clone_results/iclones/%s' %project, shell=True)
            i = 0
            for commit_id in sorted_commits:
                i += 1
                if DEBUG:
                    if i > 50:
                        break
                print(commit_id)
                print('  %.1f%%' %(i/num_commits*100))
                # checkout a specific commit
                os.chdir('../subject_systems/%s' %project)
                shellCommand('git checkout %s' %commit_id)
                os.chdir(current_dir)
                # clone detection
                if tool == 'nicad':
                    # perform clone detection by NiCad
                    shellCommand('nicad6 functions java ../subject_systems/%s' %project)
                    time.sleep(3) # to wait the clone detection to finish
                    # move the results to the result folder
                    src_path = '../subject_systems/%s_functions-blind-clones/%s_functions-blind-clones-0.30-classes.xml' %(project,project)
                    dest_path = '../clone_results/nicad/%s/%s.xml' %(project,commit_id)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dest_path)
                        # delete NiCad output files
                        subprocess.Popen('rm -rf ../src_code/%s_functions*' %project, shell=True)
                elif tool == 'iclones':
                    input_path = '../src_code/%s' %project
                    output_path = '../clone_results/iclones/%s/%s.txt' %(project,commit_id)
                    shellCommand('iclones -input %s -output %s' %(input_path,output_path))
                # clean memory

        # extract commit sequence
        commits_log_clean_path = os.path.join(config_global.COMMIT_LOG_CLEAN_PATH, '%s_logs.txt' % project)
        commits_log_df = pd.read_csv(commits_log_clean_path, names=['commit_id', 'committer', 'timestamp'], encoding="ISO-8859-1")
        
        # parse each commit's clone results
        clone_dict = DefaultDict()
    
        for commit_id in list(commits_log_df['commit_id']):
            commit_clones_list = parse_clone_result(project, commit_id)
            clone_dict[commit_id] = commit_clones_list
    
        # output results
        clone_result_purified_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH, '%s_clone_result_purified_with_paratype.json' % project)
        with open(clone_result_purified_path, 'w') as jsonfile:
            json.dump(clone_dict, jsonfile)

        project = sys.argv[1]
        tool = sys.argv[2]
        raw_genealogy_dict, end_commit_set = raw_genealogies(project, tool)
        rename_dict = renamedFiles(end_commit_set)
        combineRenamedPair(raw_genealogy_dict, rename_dict)
        commit_date_dict = loadCommitDate('../raw_data/%s_logs.txt' %project)
        fault_inducing_dict = loadFaultInducingCommits(project)
        genealogyFeatures(raw_genealogy_dict, fault_inducing_dict, commit_date_dict)

        # read in commits only related to clone groups
        group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH,
                                                     '%s_group_genealogy_distinct.csv' % (project))
        genealogy_df = pd.read_csv(group_genealogy_distinct_path)
        print(genealogy_df.shape)

        # traverse and checkout commits
        current_dir = os.getcwd()
        nicad_workdir = os.path.join(config_global.REPO_PATH, 'nicad_workdir_%s' % project)
        project_repo = os.path.join(nicad_workdir, project)
    
        print(project_repo)
        os.chdir(project_repo)
        print("cwd:", os.getcwd())
    
        cmd_create_und_db = ['git', 'config', 'core.protectNTFS', 'false']
        shellCommand(cmd_create_und_db)

        metric_columns = config_global.METRIC_COLUMNS
        cols = ['commit_id', 'clone_signiture']
        cols.extend(metric_columns)
        metrics_all_df = pd.DataFrame(columns=cols)
    
        # traverse all the start commits for the clone_group_tuple
        commit_list = ['ddd406b'] #'1a17ebc','256bf8f','8f66c7f','c32a501','7660a41','4741552','fda9aa2','1b07465','c22f209','911fc14','5d76c0b'

        for commit_id in list(genealogy_df['start_commit'].drop_duplicates()):
            print("commit_id: ", commit_id)

            # check if the corresponding metrics have been retrieved
            metrics_path = os.path.join(os.path.normpath(config_global.UDB_PATH), "%s" % project, '%s.csv' % commit_id)
            #print(metrics_path)
    
            if os.path.exists(metrics_path):
                continue
    
            # check out project repo at a specified commit to update the source repo
            cmd_checkout = ['git', 'checkout', '-f', commit_id]  # 'git checkout %s' % commit_id
            shellCommand(cmd_checkout)  # optimize: can be checked out using pydriller.Git().checkout(hash)
    
            # check if successful
            curr_commit_id = os.popen('git rev-parse --short HEAD').read()
            #n = 0
            while curr_commit_id[:len(commit_id)] != commit_id:
                print(curr_commit_id[:len(commit_id)], commit_id)
                time.sleep(1)
                #n = n + 1
                #if n == 100:
                    #sys.exit(-1)

            # get files where clones exist
            clone_files = list(get_files_by_commit(commit_id, genealogy_df))
            files_to_analyze_path = os.path.join(config_global.UDB_PATH, '%s' % project, '%s_clone_files.txt' % commit_id)
            with open(files_to_analyze_path, 'w') as fp:
                fp.write("\n".join(clone_files))
    
            # get metrics with understand tool
            build_und_project_db(commit_id, metric_columns, project, clone_files)


        # given clone_path and clone_range, retrieve clone_name
        clone_class_dict_4_clone = defaultdict(defaultdict)
        clone_class_dict_4_clone_path = os.path.join(config_global.CLONE_RESULT_PURIFIED_PATH, "%s_clone_class_dict_4_clone.pkl" % project)
        print("clone_class_dict_4_clone_path", clone_class_dict_4_clone_path)
        if os.path.exists(clone_class_dict_4_clone_path):
            print("hello exists")
            with open(clone_class_dict_4_clone_path, 'rb') as handle:
                clone_class_dict_4_clone = pickle.load(handle)
        else:
            print("not exists clone_class_dict_4_clone_path")
            clone_class_dict_4_clone = get_clone_class(project)
            with open(clone_class_dict_4_clone_path, 'wb') as handle:
                pickle.dump(clone_class_dict_4_clone, handle, protocol=pickle.HIGHEST_PROTOCOL)

        metric_on_group_df = pd.DataFrame()
    
        # loading genealogy file
        group_genealogy_distinct_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH,
                                                     '%s_group_genealogy_distinct.csv' % (project))
        genealogy_df = pd.read_csv(group_genealogy_distinct_path)
        print(genealogy_df.shape, '\n', genealogy_df.columns)

        # traverse the genealogy
        for commit_id in genealogy_df['start_commit'].unique():
    
            commit_metric_dict = get_metrics_by_commit(project, commit_id)  # metrics from understand tool
    
            commit_groups = genealogy_df[genealogy_df['start_commit'] == commit_id]['clone_group_tuple'].tolist()
        
            # traverse the groups in each commit
            for group in commit_groups:
                metric_on_group = Counter()

                clones = group.split("|")
                clone_count = len(clones)
            
                # traverse each clone in clones detected
                for clone in clones:
                    dot_java_idx = clone.rfind(".java")
                    clone = clone[0:dot_java_idx] + clone[dot_java_idx:].replace(".java", "")
                    clone = os.path.normpath(clone).replace(os.path.sep, ".")

                    if len(clone) < 3:
                        continue
                    clone_path = os.path.normpath(clone.split(":")[0])
    
                    func_name = clone_class_dict_4_clone[commit_id][clone]
    
                    clone_str = ".".join([clone_path, func_name]).strip()  # there might be spaces
    
                    clone_metrics = search_clone(commit_metric_dict, clone_str)

                    if clone_metrics is not None:
                        # only need the metrics on method level
                        clone_metrics = {key: val for key, val in clone_metrics.items() if
                                         key in config_global.METRIC_COLUMNS}
                        metric_on_group += Counter(clone_metrics)  # aggregate the metrics for clone group

                if clone_count:
                    metric_on_group_dict = dict(metric_on_group)
    
                    # get the average metric value
                    metric_on_group_dict = {k: v / clone_count for k, v in metric_on_group_dict.items()}
    
                    metric_on_group_dict.update({'clone_group_tuple': group})
    
                    metric_on_group_df = metric_on_group_df.append(metric_on_group_dict, ignore_index=True)

        group_metric_und_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric.csv' % project)
        print("group_metric_path: ", group_metric_und_path)
        metric_on_group_df.to_csv(group_metric_und_path, index=False)

        combine_und_other_metrics(metric_on_group_df)

        is_by_genealogy = True
        is_longevous = True
        is_prevalent = True
        is_buggy = True

        # loading genealogy file
        genealogy_path = os.path.join(config_global.GROUP_GENEALOGY_DISTINCT_PATH, '%s_group_genealogy_distinct.csv' % (project))
        genealogy_df = pd.read_csv(genealogy_path)
    
        # generating commit_author_dict and commit_timestamp dict
        commits_log_clean_path = os.path.join(config_global.COMMIT_LOG_CLEAN_PATH, '%s_logs.txt' % project)
        commits_log_df = pd.read_csv(commits_log_clean_path, header=None, names=['commit_id', 'committer', 'timestamp'], encoding= 'unicode_escape')
        commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['timestamp'], infer_datetime_format=True, errors='coerce')
    
        #commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['timestamp'], infer_datetime_format=True)
        commit_author_dict = dict(zip(commits_log_df.commit_id, commits_log_df.committer))
        commit_time_dict = dict(zip(commits_log_df.commit_id, commits_log_df.timestamp))
    
        genealogy_anatomized_df = anatomize_genealogy(genealogy_df, commit_time_dict, commit_author_dict)
    
        # rank by 1rd dimension - Clone LifeCycle
        rank_by_lifecycle(genealogy_anatomized_df, is_by_genealogy, config_global.threshold)
    
        # rank by 1rd dimension - Clone Prevalence
        rank_by_prevalence(genealogy_anatomized_df, config_global.threshold)
    
        # rank by 3rd dimension - Bug Proneness
        buggy_commit_list_path = os.path.join(config_global.LABEL_PATH, "buggy_commits", "%s_buggy_commits.pkl" % project)
        with open(buggy_commit_list_path, 'rb') as fp:
            buggy_commit_list = json.load(fp)
    
        rank_by_bugproneness(genealogy_anatomized_df, buggy_commit_list, config_global.threshold)
    
        # combine all the three dimensions
        decide_final_label(genealogy_anatomized_df, is_longevous, is_prevalent, is_buggy)
    
        # save labels info to file
        # label_df = genealogy_anatomized_df[['clone_group_tuple', 'start_timestamp', 'start_commit', 'churn', 'is_reusable']]
        print(genealogy_anatomized_df.columns)

        label_df = genealogy_anatomized_df[['clone_group_tuple', 'start_commit', 'n_siblings_start',
           'churn', 'n_genealogy', 'end_commit', 'n_days', 'start_timestamp', 'n_authors', 'cnt_siblings',
           'rank_by_n_genealogy', 'rank_by_n_days', 'rank_by_lifecycle',
           'rank_by_prevalence', 'n_buggy_genealogy',
           'bug_proneness', 'rank_by_bugproneness', 'is_reusable']]
        label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), '%s_3label_20230118_%s.csv' % (project, config_global.threshold))
        label_df.to_csv(label_path, index=False)
    
        print(genealogy_anatomized_df['is_reusable'].value_counts())
        #genealogy_df.drop(genealogy_df[genealogy_df['rank1'] == 1].index, inplace=True)
    
        if len(sys.argv) > 1:
            project = sys.argv[1]

        # read in metrics
        merged_metric_on_group_path = os.path.join(config_global.GROUP_METRIC_PATH, '%s_group_metric_merged.csv' % project)
        metric_on_group_df = pd.read_csv(os.path.normpath(merged_metric_on_group_path))
        print("group metric: ", metric_on_group_df.shape)
        for col_name in metric_on_group_df.columns:
            metric_on_group_df = metric_on_group_df.fillna({col_name: 0})
    
        # combine with label
        label_path = os.path.join(os.path.normpath(config_global.LABEL_PATH), '%s_3label_20230118_%s.csv' % (project, config_global.threshold))
        label_df = pd.read_csv(os.path.normpath(label_path))
    
        dataset = pd.merge(metric_on_group_df, label_df, how='inner', on='clone_group_tuple')#.drop_duplicates()
    
        raw_dataset_path = os.path.join(config_global.DATASET_PATH, '%s_raw_dataset_20230118_%s.csv' % (project, config_global.threshold))
        dataset.to_csv(raw_dataset_path, index=False)
