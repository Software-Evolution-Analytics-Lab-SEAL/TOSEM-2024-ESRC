import sys, os, re, time
from pydriller import Git
import json
import pandas as pd
import math


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

    genealogy_df['genealogy_list'] = genealogy_df['genealogy'].map(lambda gen: gen.split(';'))
    genealogy_df['commit_list'] = genealogy_df['genealogy_list'].map(lambda gen: [x.split(":")[0] for x in gen])
    genealogy_df.drop(['genealogy'], axis=1, inplace=True)
    genealogy_df['n_genealogy'] = genealogy_df['commit_list'].map(lambda x: len(x))

    # each commit in genealogy: commit_id: #updates: clone_group_sig
    genealogy_df['end_commit'] = genealogy_df['commit_list'].map(lambda x: x[-1])


    # calculate duration in terms of days
    # genealogy_df['n_days'] = list(map(lambda x, y: (commit_time_dict[y] - commit_time_dict[x]).days,
    #                                   genealogy_df['start_commit'], genealogy_df['end_commit']
    #                                   )
    #                               )  # git_repo.get_commit(commit_id).committer.email for commit_id in gen.split('-')
    
    # calucate #authors

    # genealogy_df['author_list'] = genealogy_df['genealogy_list'].map(
    #         lambda gen: set([commit_author_dict[commit.split(':', 2)[0]] for commit in gen])
    #     )
    
    # Create an empty list to store the author sets
    author_lists = []
    
    # Iterate over each genealogy list in the 'genealogy_list' column
    for gen in genealogy_df['genealogy_list']:
        
        # Create a set to store the authors for the current genealogy list
        author_set = set()
        
        # Iterate over each commit in the current genealogy list
        for commit in gen:
            # Extract the author using commit_author_dict and add it to the set
            author_set.add(commit_author_dict.get(commit.split(':', 2)[0], ""))
        
        # Append the author set to the list of author lists
        author_lists.append(author_set)
    
    # Assign the list of author lists to the 'author_list' column of the DataFrame
    genealogy_df['author_list'] = author_lists

    genealogy_df['n_authors'] = genealogy_df['author_list'].map(lambda x: len(x))
    # genealogy_df['cnt_siblings'] = genealogy_df['genealogy_list'].map(lambda gen: max([int(x.split(":")[2]) for x in gen]))

    # Create an empty list to store the maximum number of siblings for each genealogy list
    max_siblings = []
    
    # Iterate over each genealogy list in the 'genealogy_list' column
    for gen in genealogy_df['genealogy_list']:
        # Create a list to store the number of siblings for each commit
        siblings_list = []
        
        # Iterate over each commit in the current genealogy list
        for commit in gen:
            # Split the commit string and extract the number of siblings
            siblings = len((commit.split("|")))
            siblings_list.append(siblings)
        
        # Compute the maximum number of siblings for the current genealogy list
        if siblings_list:
            max_siblings.append(max(siblings_list))
        else:
            max_siblings.append(0)  # If no siblings are found, set the maximum to 0
    
    # Assign the list of maximum siblings to the 'cnt_siblings' column of the DataFrame
    genealogy_df['cnt_siblings'] = max_siblings
    # reorder the columns
    # drop duplicates
    genealogy_df['clone_group_tuple'].drop_duplicates(inplace=True)
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
        lambda x: 1 if x >= genealogy_df.n_genealogy.quantile(threshold) else 0)

    genealogy_df['rank_by_lifecycle'] = genealogy_df['rank_by_n_genealogy']


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
        lambda x: 1 if x >= genealogy_df.cnt_siblings.quantile(threshold) else 0)
    #genealogy_df['rank_by_n_authors'] = genealogy_df['rank_by_n_authors'].map({1: 'high', 0: 'low'})  # 'volvo':0 , 'bmw':1, 'audi':2} )


"""
    Functionality: Extracts bugfix commits from a DataFrame based on commit messages.

    Arguments:
    commits_log_df (DataFrame): DataFrame containing commit log data.

    Returns:
    list: List of bugfix commit IDs.
"""
def extract_bugfix_commits_by_msg(commits_log_df):
    bugfix_commit_list = []
    for index, row in commits_log_df.iterrows():
        
        bugfixes = []
        try:
            bugfixes = re.findall('(?:bug|bugs|pb|fix for|fix|fixed|fixes|resolve|solve|issue)\s*\#\s*([0-9]+)', row['commit_msg'], re.IGNORECASE)
        except:
            pass
        if len(bugfixes):
            bugfix_commit_list.append(row['commit_id'])

    return bugfix_commit_list # 



"""
    Functionality: Use szz to extract buggy commits using the specified bug fix commits for a given project,
                   and maps these to their corresponding bug fixes.

    Expected Input:
    project (str): The name of the project for which buggy commits are to be extracted.
    bugfix_commit_list (list of str): List of commit hashes that are known to be bug fixes.

    Expected Output:
    commit_bugfix_buggy_mapping (dict): A dictionary mapping each bug fix commit to its corresponding buggy commits.
"""
def extract_buggy_commits(self, project, bugfix_commit_list):
    
    project_local_repo = "../data/subject_systems/%s" % project
    from pydriller import Git
    git_repo = Git(project_local_repo)  # gr = Git('test-repos/test5')
        
    with cf.ProcessPoolExecutor() as executor:
        fn = partial(extract_buggy_commits_by_bugfix_commit, git_repo)
        results = list(executor.map(fn, bugfix_commit_list))
        
    commit_bugfix_buggy_mapping = dict(results)
    return commit_bugfix_buggy_mapping


"""
    Functionality: Identifies the commits that last modified the lines changed in a specified bugfix commit,
                   effectively tracing the origin of bugs.

    Expected Input:
    git_repo (Git.Repository): PyDriller Git repository object initialized to the project being analyzed.
    bugfix_commit (str): The hash of the bugfix commit whose buggy ancestors are to be identified.

    Expected Output:
    (tuple): A tuple where the first element is the bugfix commit hash and the second element is a list of
             commits that last modified the buggy lines (possible buggy commits).
"""
def extract_buggy_commits_by_bugfix_commit(git_repo, bugfix_commit):
    return bugfix_commit, git_repo.get_commits_last_modified_lines(git_repo.get_commit(bugfix_commit))



"""
    Functionality: Ranks the genealogy DataFrame by bug proneness.

    Expected Input:
    genealogy_df (DataFrame): DataFrame containing genealogy data.
    bugfix_commit_list (list): List of bugfix commits.
    threshold (float): Threshold value for determining the quantile.

    Expected Output: None (Updates the DataFrame in place).
"""
def rank_by_bugproneness(genealogy_df, bugfix_commit_list, threshold=0.5):
    # look into the distribution of n_genealogy
    segments = pd.cut(genealogy_df['n_genealogy'], bins=[0,2,5,100,1000])
    counts = pd.value_counts(segments, sort=True)

    bugfix_commit_list = list(set(bugfix_commit_list))

    genealogy_df['bugfix_genealogy'] = genealogy_df['commit_list'].map(lambda gen: list(set(gen).intersection(set(bugfix_commit_list))))
    genealogy_df['n_bugfix_genealogy'] = genealogy_df['bugfix_genealogy'].map(lambda gen: len(gen))

    genealogy_df['bug_proneness'] = genealogy_df.apply(lambda row:  (row['n_genealogy'] - row['n_bugfix_genealogy']) / (row['n_bugfix_genealogy'] if row['n_bugfix_genealogy'] != 0 else 1), axis=1)
    genealogy_df['rank_by_bugproneness'] = genealogy_df['bug_proneness'].map(
        lambda x: 1 if x >= genealogy_df.bug_proneness.quantile(threshold) else 0)

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

    Expected Output: Updates the DataFrame column in place.
    """
def map_label(col):
    genealogy_df[col] = genealogy_df[col].map({1: 'high', 0: 'low'})


"""
Main function to execute the program.
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        project = sys.argv[1]

    is_by_genealogy = True
    is_longevous = True
    is_prevalent = True
    is_buggy = True
    # loading genealogy file
    genealogy_df = pd.read_csv("../data/clones/%s_genealogies.csv" % project)

    # generating commit_author_dict and commit_timestamp dict
    commit_list = []
    with open('../data/commit_logs/%s_commit_log_df.csv' %project, 'r') as f:
        reader = f.read().split('\n')
        for row in reader:
            elems = row.split(',', 3)
            
            if elems and len(elems) > 1:
                commit_id = elems[0]
                committer = elems[1]
                commit_date = re.sub(r'[^0-9]', '', elems[2][:-6])
                commit_list.append([commit_id, committer, commit_date])
    commits_log_df = pd.DataFrame(commit_list, columns=['commit_id', 'committer', 'commit_date'])
    
    commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['commit_date'], infer_datetime_format=True, errors='coerce')

    #commits_log_df['timestamp'] = pd.to_datetime(commits_log_df['timestamp'], infer_datetime_format=True)
    commit_author_dict = dict(zip(commits_log_df.commit_id, commits_log_df.committer))
    commit_time_dict = dict(zip(commits_log_df.commit_id, commits_log_df.timestamp))

    genealogy_anatomized_df = anatomize_genealogy(genealogy_df, commit_time_dict, commit_author_dict)

    # rank by 1rd dimension - Clone LifeCycle
    rank_by_lifecycle(genealogy_anatomized_df, is_by_genealogy)

    # rank by 1rd dimension - Clone Prevalence
    rank_by_prevalence(genealogy_anatomized_df)

    # rank by 3rd dimension - Bug Proneness
    commit_log = pd.read_csv('../data/commit_logs/glide_commit_log_msg.csv', names=['commit_id', 'commit_msg'], quotechar="'")
    bugfix_commit_list = extract_bugfix_commits_by_msg(commit_log)
    rank_by_bugproneness(genealogy_anatomized_df, bugfix_commit_list)

    # combine all the three dimensions
    decide_final_label(genealogy_anatomized_df, is_longevous, is_prevalent, is_buggy)

    label_df = genealogy_anatomized_df[['clone_group_tuple', 'n_siblings_start',
       'n_genealogy', 'end_commit', 'n_authors', 'cnt_siblings',
       'rank_by_n_genealogy', 'rank_by_lifecycle',
       'rank_by_prevalence', 
       'bug_proneness', 'rank_by_bugproneness', 'is_reusable']]
    label_df.to_csv("../data/clones/%s_label.csv" % (project), index=False)

    print("extract labels successfully")
