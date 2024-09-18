import os, subprocess, re, json, sys, time
import pandas as pd
from collections import defaultdict, OrderedDict


"""
    Functionality: Executes a shell command and returns the output and error messages.
    
    Expected Input:
    command_str (str): The shell command to be executed.
    
    Expected Output:
    cmd_out (bytes): The output of the shell command.
    cmd_err (bytes): The error messages (if any) produced by the shell command.
"""
def shellCommand(command_str):
    cmd = subprocess.Popen(command_str.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out, cmd_err = cmd.communicate()
    return cmd_out


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
    clone_info = re.findall(
        r'file\=\"(.+?)\"\s+startline\=\"([0-9]+)\"\s+endline\=\"([0-9]+)\"\s+pcid=\"([0-9]+)\">\s+(.*)\)',
        clone_info_str[:idx_right_parenthesis + 1], 
        re.S)
    
    if len(clone_info) > 0:
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
        func_paras = extract_para_types(func_header[1].strip())
        
        return [file_path, startline + '-' + endline, func_name + "(" + func_paras + ")"]
    else:
        #breakpoint()
        print("no clone classes find: ", commit_id)
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
    commit_clone_result_path = '../data/clones/%s/%s.xml' % (project, commit_id)
    print('processing the commit: ', commit_id, '...')
    # breakpoint()
    if os.path.exists(commit_clone_result_path):
        with open(commit_clone_result_path, 'r', encoding="ISO-8859-1") as f:
        #with open(commit_clone_result_path, 'r', encoding='unicode_escape') as f:
            reader = f.read()
            
            # extract a pair of clones
            group_list = re.findall(
                r'<class classid=\"[0-9]+\" nclones=\"[0-9]+\" nlines=\"[0-9]+\" similarity=\"[0-9]+\">(.+?)</class>',
                reader, re.DOTALL)
            
            for group in group_list:  # one class
                # extract clone strings
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
    
    return result_list


"""
Main function to execute the program.
"""
if __name__ == '__main__':
    if len(sys.argv) > 1:
        project = sys.argv[1]

    #project = 'glide'  # func_paras = extract_para_types(func_header[1].strip()) IndexError: list index out of range

    # extract commit sequence
    commit_list = list()

    # the commit logs can be retrieved directly through "git log" command.
    with open('../data/commit_logs/%s_commit_log_df.csv' % project, 'r') as f:
        reader = f.read().split('\n')
        for index, row in enumerate(reader):
            if index == 0:  # This checks if the current row is the first line
                continue    # Skip processing for the first line
            elems = row.split(',', 3)
            
            if elems and len(elems) > 1:
                commit_id = elems[0]
                commit_date = re.sub(r'[^0-9]', '', elems[2][:-6])
                commit_list.append([commit_id, commit_date])
                
    commits_log_df = pd.DataFrame(commit_list, columns=['commit_id', 'commit_date'])
    
    # parse each commit's clone results
    clone_dict = OrderedDict()

    for commit_id in list(commits_log_df['commit_id']):
        commit_clones_list = parse_clone_result(project, commit_id)
        
        clone_dict[commit_id] = commit_clones_list

    # output results
    with open('../data/clones/%s_clone_result_purified_with_paratype.json' % project, 'w') as jsonfile:
        json.dump(clone_dict, jsonfile)
