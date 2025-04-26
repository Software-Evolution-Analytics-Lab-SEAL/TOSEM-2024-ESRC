from __future__ import division
import sys, os, re, subprocess, shutil
import pandas as pd


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
                
    df = pd.DataFrame(commit_list, columns=['commit_id', 'commit_date'])
    sorted_commits = list(df.sort_values('commit_date')['commit_id'])
    return sorted_commits


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
                subprocess.Popen('rm -rf ../data/subject_systems/%s_functions*' %project, shell=True)
                subprocess.Popen('mkdir -p ../data/clones/%s' %project, shell=True)
            elif tool == 'iclones':
                subprocess.Popen('mkdir -p ../data/clones/%s' %project, shell=True)
            i = 0
            for commit_id in sorted_commits:
                i += 1
                if DEBUG:
                    if i > 50:
                        break
                
                print('  %.1f%%' %(i/num_commits*100))
                # checkout a specific commit
                os.chdir('../data/subject_systems/%s' %project)
                shellCommand('git checkout %s' %commit_id)
                os.chdir(current_dir)
                # clone detection
                if tool == 'nicad':
                    # perform clone detection by NiCad
                    shellCommand('nicad6 functions java ../data/subject_systems/%s' % project)
                    # move the results to the result folder
                    src_path = '../data/subject_systems/%s_functions-blind-clones/%s_functions-blind-clones-0.30-classes-withsource.xml' %(project,project)
                    dest_path = '../data/clones/%s/%s.xml' %(project,commit_id)
                    if os.path.exists(src_path):
                        shutil.move(src_path, dest_path)
                        # delete NiCad output files
                        subprocess.Popen('rm -rf ../src_code/%s_functions*' %project, shell=True)
                elif tool == 'iclones':
                    input_path = '../src_code/%s' %project
                    output_path = '../data/clones/iclones/%s/%s.txt' %(project,commit_id)
                    shellCommand('iclones -input %s -output %s' % (input_path,output_path))
                # clean memory
                
    
