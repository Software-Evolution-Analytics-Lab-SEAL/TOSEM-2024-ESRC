import logging
import subprocess
import sys,os
import platform
import time
import pandas as pd
import shutil
pd.set_option('display.max_columns', None)
sys.path.append('..')
expanded_path = os.path.expanduser('~')

"""
    Functionality: Executes a shell command and captures the output and errors.

    Expected Input:
    command_str (list): A list containing the shell command and its arguments.

    Expected Output:
    cmd_out (bytes): The standard output from the command.
    cmd_err (bytes): The standard error from the command.
    """
def shellCommand(command_str):
    cmd = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_str = " ".join(command_str)
    cmd_out, cmd_err = cmd.communicate()
    return cmd_out, cmd_err


"""
    Functionality: Builds the Understand project database for a specific commit.

    Expected Input:
    commit_id (str): The ID of the commit.
    metric_columns (list): A list of metric columns to be analyzed.
    project (str): The name of the project.
    clone_files (list): A list of clone files.

    Expected Output:
    built understand database
    """
def build_und_project_db(commit_id, metric_columns, project, clone_files):
    # run understand cli to construct the project understand db
    # create db
    
    und_commit_db = expanded_path + '/topic1-replication-package/data/clones/udb/%s/%s.und' % (project, commit_id)
    if os.path.exists(und_commit_db):
        shutil.rmtree(und_commit_db, ignore_errors=True)

    # create und db
    cmd_create_und_db = ['und', '-db', und_commit_db, 'create', '-languages', 'Java']
    shellCommand(cmd_create_und_db)

    # add all files into db corresponding to the current commit
    for clone_file in clone_files:
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
            #print(clone_path)
            #clone_str = eval(repr(clone.split("-")[0].replace("'", "").strip())).replace('\\\\', '\\')
            #clone_path = clone_path.replace("%s\\" % project, "") # 去掉前面的project 名称
            # clone = os.path.normpath(clone.replace(".java", "")).replace(os.path.sep, ".")
            # clone_path = os.path.normpath(clone.split(":")[0])
            if len(clone):
                files_by_commit.add(clone_path)
    
    return files_by_commit


"""
Main function to execute the program.
"""
if __name__ == '__main__':
    if len(sys.argv) > 1:
        project = sys.argv[1]
    
    os.makedirs('../data/clones/udb/%s' % project, exist_ok=True)
    # read in commits only related to clone groups
    genealogy_df = pd.read_csv('../data/clones/%s_genealogies.csv' % project)
    # traverse and checkout commits
    current_dir = os.getcwd()
    os.chdir('../data/subject_systems/%s' % project)
    print("cwd:", os.getcwd())

    cmd_create_und_db = ['git', 'config', 'core.protectNTFS', 'false']
    shellCommand(cmd_create_und_db)
    
    cols = ['commit_id', 'clone_signiture', "CountInput"
                 ,"CountLine"
                 ,"CountLineBlank"
                 ,"CountLineCode"
                 ,"CountLineCodeDecl"
                 ,"CountLineCodeExe"
                 ,"CountLineComment"
                 ,"CountOutput"
                 ,"CountPath"
                 ,"CountSemicolon"
                 ,"CountStmt"
                 ,"CountStmtDecl"
                 ,"CountStmtEmpty"
                 ,"CountStmtExe"
                 ,"Cyclomatic"
                 ,"CyclomaticModified"
                 ,"CyclomaticStrict"
                 ,"Essential"
                 ,"EssentialStrictModified"
                 ,"Knots"
                 ,"RatioCommentToCode"
                 ,"MaxEssentialKnots"
                 ,"MaxNesting"
                 ,"MinEssentialKnots"
                 ,"SumCyclomatic"
                 ,"SumCyclomaticModified"
                 ,"SumCyclomaticStrict"
                 ,"SumEssential"]
    # traverse all the start commits for the clone_group_tuple
    for commit_id in list(genealogy_df['start_commit'].drop_duplicates()): # da3acb16

        # check if the corresponding metrics have been retrieved
        metrics_path = expanded_path + '/topic1-replication-package/data/clones/udb/%s/%s.csv' % (project, commit_id)
        # check out project repo at a specified commit to update the source repo
        cmd_checkout = ['git', 'checkout', '-f', commit_id]  # 'git checkout %s' % commit_id
        shellCommand(cmd_checkout)  # optimize: can be checked out using pydriller.Git().checkout(hash)

        # get files where clones exist
        clone_files = list(get_files_by_commit(commit_id, genealogy_df))

        # get metrics with understand tool
        metric_columns = ["CountInput"
                 ,"CountLine"
                 ,"CountLineBlank"
                 ,"CountLineCode"
                 ,"CountLineCodeDecl"
                 ,"CountLineCodeExe"
                 ,"CountLineComment"
                 ,"CountOutput"
                 ,"CountPath"
                 ,"CountSemicolon"
                 ,"CountStmt"
                 ,"CountStmtDecl"
                 ,"CountStmtEmpty"
                 ,"CountStmtExe"
                 ,"Cyclomatic"
                 ,"CyclomaticModified"
                 ,"CyclomaticStrict"
                 ,"Essential"
                 ,"EssentialStrictModified"
                 ,"Knots"
                 ,"RatioCommentToCode"
                 ,"MaxEssentialKnots"
                 ,"MaxNesting"
                 ,"MinEssentialKnots"
                 ,"SumCyclomatic"
                 ,"SumCyclomaticModified"
                 ,"SumCyclomaticStrict"
                 ,"SumEssential"
                ]
        build_und_project_db(commit_id, metric_columns, project, clone_files)

        print("finish processing commit_id: ", commit_id)
        
