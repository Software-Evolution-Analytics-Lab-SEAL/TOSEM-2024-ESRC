# relate metrics to clone group
import os, sys, json, pickle, time, requests, random, re
import pandas as pd
from collections import Counter
from collections import defaultdict
tree = lambda: defaultdict(tree)
import networkx as nx
from tqdm import tqdm
import concurrent.futures as cf
import git
from github import Github
from git import Repo
from pydriller import Git
from difflib import SequenceMatcher

SUBJECT_SYSTEMS_ALL = {
'Anki-Android': "ankidroid/Anki-Android",
'che': "eclipse/che",                               # beluga_scratch
'checkstyle': "checkstyle/checkstyle",              # ~ done
'druid': "apache/druid",                            #graham_clone2api_1_slurm-53093081
'framework': "vaadin/framework",                    #cedar_scratch
'gatk': "broadinstitute/gatk",                      #cedar_scratch
'graylog2-server': "Graylog2/graylog2-server",      #cedar_scratch
'grpc-java': "grpc/grpc-java",                      #cedar_scratch
'hazelcast': "hazelcast/hazelcast",               #beluga_scratch
'jabref': "JabRef/jabref",                          #beluga_scratch - unicode error
'k': "runtimeverification/k",                       #jzlarge done
'k-9': "k9mail/k-9",                                #jzlarge 12%
'mage': "magefree/mage",                            #beluga_scratch
'minecraftForge': "MinecraftForge/MinecraftForge",  #beluga_scratch - done with error - 7890/7891
'molgenis': "molgenis/molgenis",                    #graham_scratch
'muikku': 'otavanopisto/muikku',                    #graham_scratch
'nd4j': "deeplearning4j/nd4j",                      #graham_scratch # git repo changed
'netty': "netty/netty",                             #graham_scratch - oom on graham
'openhab1-addons': "openhab/openhab1-addons",       #graham_scratch
'pinpoint': "pinpoint-apm/pinpoint",                #graham_scratch - killed
'presto': "prestodb/presto",                        #  <= "facebook/presto",   #beluga_scratch
'product-apim': "wso2/product-apim",                #beluga_scratch
'realm-java': "realm/realm-java",                   #beluga_scratch - done on belugals - 8805/8805
'reddeer': "jboss-reddeer/reddeer",                 #beluga_scratch - done on jzsmall - 1612/1612 done on beluga
'RxJava': "ReactiveX/RxJava",                       #beluga_scratch - done on beluga - 5945/5945
'smarthome': "eclipse-archived/smarthome",          #cedar_scratch - # git repo archived
'Terasology': "MovingBlocks/Terasology",            #cedar_scratch
'wildfly-camel': "wildfly-extras/wildfly-camel",    #cedar_scratch  - done on jzsmall - 1677/1677  done on cedarl - local
'XChange': "knowm/XChange",                     #cedar_scratch
'xp': "enonic/xp",                                  #cedar_scratch
'zaproxy': "zaproxy/zaproxy",                       #~ done
'spring-boot': "spring-projects/spring-boot",

# c projects
"betaflight": "betaflight/betaflight",
"cleanflight": "cleanflight/cleanflight",
"collectd": "collectd/collectd",
"fontforge": "fontforge/fontforge",
"FreeRDP": "FreeRDP/FreeRDP",
"inav": "iNavFlight/inav",
"john": "openwall/john",
"libgit2": "libgit2/libgit2",
"lxc": "lxc/lxc",
"micropython": "micropython/micropython",
"mpv": "mpv-player/mpv",
"netdata": "netdata/netdata",
"ompi": "open-mpi/ompi",
"radare2": "radareorg/radare2",
"redis": "redis/redis",
"RIOT": "RIOT-OS/RIOT",
"systemd": "systemd/systemd",
"zfs": "openzfs/zfs",

# young
'SoftEtherVPN': 'SoftEtherVPN/SoftEtherVPN', 
'phpredis': 'phpredis/phpredis', 
'nodemcu-firmware': 'nodemcu/nodemcu-firmware', 
'WinObjC': 'microsoft/WinObjC', 
'AdAway': 'AdAway/AdAway', 
'wasm-micro-runtime': 'bytecodealliance/wasm-micro-runtime', 
'nDPI': 'ntop/nDPI', 
'nng': 'nanomsg/nng', 
'premake-core': 'premake/premake-core', 
'InfiniTime': 'InfiniTimeOrg/InfiniTime', 
'pgbackrest': 'pgbackrest/pgbackrest',  # c
'inspektor-gadget': 'inspektor-gadget/inspektor-gadget',  # c
'samtools': 'samtools/samtools', 
'arduino-pico': 'earlephilhower/arduino-pico', 
'open5gs': 'open5gs/open5gs', 
'glide': 'bumptech/glide',  # java
'PocketHub': 'pockethub/PocketHub', 
'shardingsphere-elasticjob': 'apache/shardingsphere-elasticjob', 
'swagger-core': 'swagger-api/swagger-core', 
'HMCL': 'huanghongxun/HMCL', 
'YCSB': 'brianfrankcooper/YCSB', 
'maxwell': 'zendesk/maxwell', 
'spock': 'spockframework/spock', 
'spring-data-elasticsearch': 'spring-projects/spring-data-elasticsearch', 
'Rajawali': 'Rajawali/Rajawali', 
'appinventor-sources': 'mit-cml/appinventor-sources', 
'shenyu': 'apache/shenyu', 
'baritone': 'cabaletta/baritone', 
'light-4j': 'networknt/light-4j', 
'firebase-android-sdk': 'firebase/firebase-android-sdk',

# middle-aged
'hashcat':'hashcat/hashcat',
'sway': 'swaywm/sway',				
'lvgl':'lvgl/lvgl',				
'libevent':'libevent/libevent',	
'zfs':'openzfs/zfs',				    
'i3':'i3/i3',	            
'libvips':'libvips/libvips',	
'poco':'pocoproject/poco',	
'klipper':'Klipper3d/klipper',
'pygame':'pygame/pygame',	
'flatpak':'flatpak/flatpak',	
'stellar-core':'stellar/stellar-core',			
'surge':'surge-synthesizer/surge',	
'dynamorio':'DynamoRIO/dynamorio',	
'czmq':'zeromq/czmq',				
'RxJava':'ReactiveX/RxJava',		
'skywalking':'apache/skywalking',	
'mockito':'mockito/mockito',			
'realm-java':'realm/realm-java',		
'zaproxy':'zaproxy/zaproxy',			
'grpc-java':'grpc/grpc-java',				        
'MinecraftForge':'MinecraftForge/MinecraftForge',	
'Peergos':'Peergos/Peergos',				            
'jooby':'jooby-project/jooby',				        
'gatk':'broadinstitute/gatk',				        
'Mekanism':'mekanism/Mekanism',		    	
# 'halo':'halo-dev/halo',		# Java 48.0%	            
'janusgraph':'JanusGraph/janusgraph',		
'iceberg':'apache/iceberg',

# old
'netdata':'netdata/netdata',		# 		
'redis': 'redis/redis',						
'mpv':'mpv-player/mpv',						
'radare2':'radareorg/radare2',				
 #'micropython':'micropython/micropython',		
'systemd':'systemd/systemd',				
# 'libgit2':'libgit2/libgit2',					
# 'FreeRDP':'FreeRDP/FreeRDP',					
'john':'openwall/john',					
'betaflight':'betaflight/betaflight',		
# 'fontforge':'fontforge/fontforge',			
#'RIOT':'RIOT-OS/RIOT', 					
'lxc':'lxc/lxc',								
'collectd':'collectd/collectd',			
'cleanflight':'cleanflight/cleanflight',		
# 'inav':'iNavFlight/inav',					
# 'ompi':'open-mpi/ompi',					
# 'spring-boot':'spring-projects/spring-boot',	# java
'netty':'netty/netty',						
'presto':'prestodb/presto',					
'pinpoint':'pinpoint-apm/pinpoint',		
'druid':'apache/druid',						
'checkstyle':'checkstyle/checkstyle',		
#'graylog2-server':'Graylog2/graylog2-server',	
'hazelcast':'hazelcast/hazelcast',			
'XChange':'knowm/XChange',					
'Terasology':'MovingBlocks/Terasology',		
'jabref':'JabRef/jabref',					
'framework':'vaadin/framework',				
#'mage':'magefree/mage',
}

commit_modification_dict = dict()
blame_cache = defaultdict(set)
def init_commit_modification_dict(project):
    commit_list = []
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
    commit_log_df = pd.DataFrame(commit_list, columns=['commit_id', 'commit_date'])
    
    project_repo = Git("../data/subject_systems/%s" % project)
    for commit_id in commit_log_df['commit_id']:
        commit_modification_dict[commit_id] = list()
        commit_modified_files = list(filter(lambda x: (x.old_path and x.old_path.endswith(f'.java')) or
                                                       (x.new_path and x.new_path.endswith(f'.java')),
                                             project_repo.get_commit(commit_id).modified_files))
        for modified_java_file in commit_modified_files:
            commit_modification_dict[commit_id].append([modified_java_file.old_path, modified_java_file.new_path, modified_java_file.diff])

"""
    Functionality: Retrieves clone classes for a given project.

    Expected Input:
    project (str): The name of the project.

    Expected Output:
    clone_class_dict_4_clone (defaultdict): A nested dictionary containing clone classes.
    """
def get_clone_class(project):
    clone_class_dict_4_clone = defaultdict(defaultdict)
    project_clone_result_purified_path = '../data/clones/%s_clone_result_purified_with_paratype.json' % project
    # load clone classes
    with open(project_clone_result_purified_path, 'r') as clone_jsonfile:
        clone_result_json = json.load(clone_jsonfile)

        for commit_id in clone_result_json:
            # filter out test functions
            for clone_group in clone_result_json[commit_id]:
                for clone in clone_group:
                    dot_java_idx = clone[0].rfind(".java")
                    #print("clone: ", clone)
                    clone[0] = clone[0][0:dot_java_idx] + clone[0][dot_java_idx:].replace(".java", "")
                    clone[0] = os.path.normpath(clone[0]).replace(os.path.sep, ".")

                    #clone[0] = os.path.normpath(clone[0].split(".java")[0]).replace(os.path.sep, '.')  # remove .java and replace / with .
                    if clone[0].lower().find('test') == -1:  # filter out test methods
                        clone_signiture = ':'.join(clone[:2])  # clone[2] is the function name

                        clone_class_dict_4_clone[commit_id][clone_signiture] = clone[2]
    #print(clone_class_dict_4_clone)
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
    
    commit_metric_path = '../data/clones/udb/%s/%s.csv' % (project, commit_id)
    if not os.path.exists(commit_metric_path):
        return
    
    # filter out non-methods
    commit_metric_df = pd.read_csv(commit_metric_path)
    commit_metric_df = commit_metric_df[commit_metric_df['Kind'].str.contains('method', case=False) |
                                        commit_metric_df['Kind'].str.contains('function', case=False) |
                                        commit_metric_df['Kind'].str.contains('procedure', case=False) |
                                        commit_metric_df['Kind'].str.contains('constructor', case=False)
                                        ]

    if 'Kind' in commit_metric_df.columns:
        commit_metric_df.drop(['Kind'], axis=1, inplace=True)
    if 'File' in commit_metric_df.columns:
        commit_metric_df.drop(['File'], axis=1, inplace=True)

    # process the function signiture column
    pattern = "\\(.*?\\)"
    #commit_metric_df['Name'] = commit_metric_df['Name'].str.replace(pattern, '')

    #commit_metric_df['Name'] = commit_metric_df['Name'].str.replace('\.[a-zA-Z0-9_]+\.\.', '.') # remove (Anon_1)
    commit_metric_df['Name'] = commit_metric_df['Name'].str.replace('.\(Anon_[0-9]+\).', '.', regex=True)  # remove (Anon_1)

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
    methods_in_samefile = []
    for key in commit_metric_dict:
        methods_in_samefile.append(key)
        idx = key.find('(')
        key_without_class_info = key[:idx].split(".")
        key_without_class_info.pop(-2)
        key_without_class_info_str = '.'.join(key_without_class_info)
        key_path = '.'.join(key.split('.')[:-1])
        key_funcname = "".join(key.split('.')[-1:])
        
        # some method from understand tool has class info
        if key in clone_str:
            return commit_metric_dict.get(key)
        elif (key_without_class_info_str in clone_str) and (key_funcname in clone_str):
            return commit_metric_dict.get(key)
        elif key_funcname in clone_str:
            return commit_metric_dict.get(key)
        elif key_without_class_info_str in clone_str:
            return commit_metric_dict.get(key)
        elif key_path in clone_str:
            return commit_metric_dict.get(key)
        elif SequenceMatcher(None, key, clone_str).ratio() >= 0.7:
            return commit_metric_dict.get(key)

    if commit_metric_dict:
        rkey = random.choice(list(commit_metric_dict.keys()))
        return commit_metric_dict.get(rkey)
    
    if len(methods_in_samefile):
        def get_averages(list_of_methodmetrics_in_samefile):
            # Create a dictionary to store the sums
            sums = {}
            for d in list_of_methodmetrics_in_samefile:
                for key, value in d.items():
                    # Add the value to the running total for this key
                    if key in sums:
                        sums[key] += value
                    else:
                        sums[key] = value
        
            # Calculate averages and store in a new dictionary
            averages = {}
            for key, value in sums.items():
                averages[key] = value / len(list_of_methodmetrics_in_samefile)
            return averages
        
        average_metric = get_averages([commit_metric_dict.get(key) for key in methods_in_samefile])
        return average_metric

    return None


def get_contributors_by_clonefile(github_base_url, commit_id, clone_file_path):
    from posixpath import join
    clone_contributor_url = join(github_base_url, "contributors-list", commit_id, clone_file_path.replace('\\', '/'))
    max_retries, retry_delay = 5, 2
    # default_header = requests.sessions.Session().default_headers
    header = headers
    for attempt in range(max_retries):
        try:
            github_html = requests.get(clone_contributor_url, headers=header).text
            soup = BeautifulSoup(github_html, "html.parser")
            contributors_block = soup.find_all('a', {"class": "Link--primary no-underline"})

            contributors = set()
            for contributor in contributors_block:
                contributor_id = contributor.get('href')
                contributors.add(contributor_id)
            return contributors
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.TooManyRedirects) as e:
            # Exception resolution logic for ConnectionError, Timeout, and TooManyRedirects
            print("An error occurred:", str(e))
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)
            elif header == headers: # try add headers
                print("Maximum number of retries reached. Exiting.")
                header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                       "Content-Type": "application/json"}
                attempt -= 1 # give another retry with different header, be careful, this might cause unlimited loop
            else:
                return None


def process_row(row, repo):
        start_commit = row['start_commit']
        clone_group_tuple = row['clone_group_tuple']
        genealogy = row['genealogy']
    
        group_contributors = set()
        commits = genealogy.split(';')

        # filter commit
        commits = [commit for commit in commits if commit.split(":", 1)[0] in list(commit_modification_dict.keys())]
        if len(commits) > 1000:
            commits = random.sample(commits, 1000)
    
        with cf.ThreadPoolExecutor(max_workers=100) as executor:
            future_results = {executor.submit(process_commit, commit, repo): commit for commit in commits}
    
            for future in cf.as_completed(future_results):
                group_contributors.update(future.result())
    
        return group_contributors


def get_contributors_by_clone(repo, commit_id, clone_file_path, start_line, end_line):
    cache_key = f"{commit_id}:{clone_file_path}:{start_line}:{end_line}"
    
    # Check if result is cached
    if cache_key in blame_cache:
        return blame_cache[cache_key]
    
    authors = set()
    try:
        # Run blame command
        blame_info = repo.git.blame("-e", f"-L {start_line},{end_line}", commit_id, '--', clone_file_path).splitlines()
        for line in blame_info:
            author_email = line.split('(')[1].split(' ')[0].strip()[1:-1].strip()
            authors.add(author_email)
            #authors[author] = authors.get(author, 0) + 1 # the #lines the author made
    except Exception as e:
        return set()
    # Cache the result
    blame_cache[cache_key] = authors
    return authors


def process_commit(commit, repo):
    commit_id, _, _, clone_group_variant = commit.split(":", 3)
    clone_siblings = clone_group_variant.split("|")
    contributors_for_commit = set()

    for clone in clone_siblings:
        #print("clone_file_path: ", clone)
        clone_file_path = clone.split(":")[0]
        start_line, end_line = clone.split(":")[1].split('-')
        clone_contributors = get_contributors_by_clone(repo, commit_id, clone_file_path, start_line, end_line)
        contributors_for_commit.update(clone_contributors)
     
    return contributors_for_commit


def get_group_contributors(project, repo, genealogy_df):
    group_contributors_dict = dict()
    with cf.ProcessPoolExecutor(max_workers=10) as process_executor:
        future_to_row = {process_executor.submit(process_row, row, repo): row for index, row in genealogy_df.iterrows()}

        group_contributors_dict = {} 
        for future in tqdm(cf.as_completed(future_to_row), total=len(future_to_row)):
            row = future_to_row[future]
            group_contributors = future.result()
            clone_group_tuple = row['clone_group_tuple']
            group_contributors_dict[clone_group_tuple] = list(group_contributors)
    
    return group_contributors_dict
        

def get_github_contributors_email_name(project):
    project_local_repo = Repo('../data/subject_systems/%s' % project)
    committer_set = set()
    for commit in project_local_repo.iter_commits():
        author_email = commit.author.email
        author_name = commit.author.name
        committer_email = commit.committer.email
        committer_name = commit.committer.name
        
        committer_set.add((committer_email, committer_name))
        committer_set.add((author_email, author_name))
    email_name_map_df = pd.DataFrame(list(committer_set), columns=['email', 'name'])
    return email_name_map_df


def get_github_contributors(project):
    # Initialize Github object using your access token
    for token in ['put your tokens here']:
        print("using Github token, check if the token expires!")
        try:
            g = Github(token)
            repo = g.get_repo(SUBJECT_SYSTEMS_ALL[project])
            
            # Fetch the list of contributors
            contributors = repo.get_contributors()
            contributor_set = set()
            # Loop through PaginatedList of NamedUser (contributors) and print their login name
            for contributor in contributors:
                if contributor.name:
                    contributor_set.add((contributor.name, contributor.login, contributor.email, contributor.followers, 
                                         contributor.created_at, contributor.contributions))
            contributor_df = pd.DataFrame(list(contributor_set), 
                                          columns=['contributor_name', 'contributor_login', 
                                                   'contributor_email', 'cnt_followers', 
                                                   'contributor_createat', 'contributions'])
            contributor_df['cnt_followers'] = contributor_df['cnt_followers'].astype(int)
            contributor_df['contributions'] = contributor_df['contributions'].astype(int)
            contributor_df['contributor_name'] = contributor_df['contributor_name'].str.strip()
            email_name_map_df = get_github_contributors_email_name(project)
            email_merged = contributor_df.merge(email_name_map_df, how='left', left_on='contributor_email', right_on='email')
            name_merged = contributor_df.merge(email_name_map_df, how='left', left_on='contributor_name', right_on='name')
            merged_df = pd.merge(email_merged, name_merged, on='contributor_login', suffixes=('_df1', '_df2'))
            def combine_emails(row):
                emails = []
                for col in ['contributor_email_df1', 'email_df1', 'contributor_email_df2',  'email_df2']:
                    email = row[col]
                    if pd.notna(email):
                        emails.append(email)
                return list(set(emails))
            
            def combine_names(row): # 'contributor_login', 'cnt_followers_df1', 'cnt_followers_df2'
                names = []
                for col in ['contributor_name_df1', 'name_df1', 'contributor_name_df2',  'name_df2']:
                    name = row[col]
                    if pd.notna(name):
                        names.append(name)
                return list(set(names))
            merged_df['combined_emails'] = merged_df.apply(combine_emails, axis=1)
            merged_df['combined_names'] = merged_df.apply(combine_names, axis=1)
            merged_df = merged_df[['contributor_login', 'cnt_followers_df1', 'contributor_createat_df1', 'contributions_df1', 'combined_emails', 'combined_names']]
            merged_df.rename(columns={'contributor_login': 'contributor_login', 
                               'cnt_followers_df1': 'cnt_followers',
                               'contributor_createat_df1': 'contributor_createat', 
                               'contributions_df1': 'contributions',
                               'combined_emails': 'emails',
                               'combined_names': 'names'}, inplace=True)
            
            # Function to merge lists and remove duplicates
            def merge_and_unique(s):
                merged = []
                for lst in s:
                    merged.extend(lst)
                return list(set(merged))  # Remove duplicates using set()
            
            # Group by 'contributor_login' and apply the function
            merged_df_grouped = merged_df.groupby('contributor_login').agg({
                'cnt_followers': 'first',  # or 'max'/'min'/'mean' or any other aggregation
                'contributor_createat': 'first',
                'contributions': 'first',
                'emails': merge_and_unique,
                'names': merge_and_unique  # or 'max'/'min'/'mean' or any other aggregation
            }).reset_index()
            return merged_df_grouped
        except Exception as err:
            continue


def get_path_metric_by_jaccard_similarity(file_paths):
    def jaccard_similarity(a, b):
        a, b = set(a.split(os.path.sep)), set(b.split(os.path.sep))
        return len(a & b) / len(a | b)
    total_similarity = sum(jaccard_similarity(a, b) for a in file_paths for b in file_paths)
    total_count = len(file_paths)**2
    
    average_similarity = total_similarity / total_count
    diversity_metric = 1 - average_similarity
    #print(f"!!!{file_paths} jaccard {diversity_metric}")
    return diversity_metric


def generate_other_metrics(project):
    ## read in commits only related to clone groups
    genealogy_df = pd.read_csv('../data/clones/%s_genealogies.csv' % project)
    print(project, genealogy_df.shape, genealogy_df.columns)
    # Path to your local repository
    repo = git.Repo('../data/subject_systems/%s' % project)
    group_contributors_dict = get_group_contributors(project, repo, genealogy_df)
    
    project_contributor_df = get_github_contributors(project)
    # project_contributor_df['contributor_createat'] = pd.to_datetime(project_contributor_df['contributor_createat'])
    # project_contributor_df['contributor_years_experience'] = (pd.Timestamp('2023-05-01') - project_contributor_df['contributor_createat']).dt.days
    # project_contributor_df['contributor_years_experience'] = project_contributor_df['contributor_years_experience'].astype(int)
    project_contributor_df['contributions'] = project_contributor_df['contributions'].astype(int)
    for index, row in tqdm(genealogy_df[['clone_group_tuple', 'start_commit']].iterrows(), total=genealogy_df.shape[0]):
        clone_siblings = row['clone_group_tuple'].split("|")
        cnt_clone_siblings = len(clone_siblings)

        cnt_group_paras = 0
        # print(clone_siblings)
        for clone in clone_siblings:
            if len(clone) < 3:
                continue
            
            # func_name = clone_class_dict_4_clone[row['start_commit']][clone]
            #func_paras = re.findall(r"[(](.*?)[)]", func_name)[0]
            cnt_func_paras = len(list(filter(None, clone.split(","))))
            cnt_group_paras += cnt_func_paras

        genealogy_df.loc[index, 'cnt_group_paras'] = int(cnt_group_paras / cnt_clone_siblings)
        genealogy_df.loc[index, 'cnt_clone_siblings'] = cnt_clone_siblings
        # get common_path of clone siblings
        clone_siblings_paths = [clone_sibling.split(':')[0] for clone_sibling in clone_siblings]
        genealogy_df.loc[index, 'path_jaccard_similarity'] = get_path_metric_by_jaccard_similarity(clone_siblings_paths)
        
        cnt_group_followers, cnt_group_experience, cnt_group_contributions = 0, 0, 0
        group_contributors = group_contributors_dict[row['clone_group_tuple']]
        genealogy_df.loc[index, 'cnt_distinct_contributors'] = len(group_contributors)

        for contributor_email in group_contributors:
            try:
                # Find cnt_followers based on contributor_name
                matched = project_contributor_df[project_contributor_df['emails'].apply(lambda x: contributor_email in x)]
                if matched.empty:
                    email_name_map_df = get_github_contributors_email_name(project)
                    contributor_name_matched_df = email_name_map_df[email_name_map_df['email']==contributor_email]
                    if contributor_name_matched_df.empty:
                        cnt_group_followers += 0
                        # cnt_group_experience += 0
                        cnt_group_contributions += 0
                    else:
                        for contributor_name in contributor_name_matched_df['name']:
                            name_matched = project_contributor_df[project_contributor_df['names'].apply(lambda x: contributor_name in x)]
                            if name_matched.empty:
                                cnt_group_followers += 0
                                cnt_group_experience += 0
                                cnt_group_contributions += 0
                            else:
                                contributor_login = name_matched['contributor_login'].tolist()[0]
                                cnt_followers = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['cnt_followers'].tolist()[0]
                                cnt_experience = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributor_years_experience'].tolist()[0]
                                cnt_contributions = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributions'].tolist()[0]
                                cnt_group_followers += int(cnt_followers)
                                # cnt_group_experience += int(cnt_experience)
                                cnt_group_contributions += int(cnt_contributions)
                                break
                else:
                    contributor_login = matched['contributor_login'].tolist()[0]
                    cnt_followers = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['cnt_followers'].tolist()[0]
                    # cnt_experience = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributor_years_experience'].tolist()[0]
                    cnt_contributions = project_contributor_df[project_contributor_df['contributor_login'] == contributor_login]['contributions'].tolist()[0]
                    cnt_group_followers += int(cnt_followers)
                    # cnt_group_experience += int(cnt_experience)
                    cnt_group_contributions += int(cnt_contributions)
            except Exception as err:
                print(err)
        genealogy_df.loc[index, 'cnt_group_followers'] = int(cnt_group_followers / cnt_clone_siblings)
        # genealogy_df.loc[index, 'cnt_group_experience'] = int(cnt_group_experience / cnt_clone_siblings)
        genealogy_df.loc[index, 'cnt_group_contributions'] = int(cnt_group_contributions / cnt_clone_siblings)

    genealogy_df.drop(['start_commit', 'genealogy'], axis=1, inplace=True)
    return genealogy_df
    

"""
    Functionality: Combines Understand metrics with other metrics for clone groups.

    Expected Input:
    und_metric_on_group_df (DataFrame): DataFrame containing Understand metrics for clone groups.
    project (str): The name of the project.

    Expected Output:
    None
"""
def combine_und_other_metrics(und_metric_on_group_df):
    other_metric_on_group_df = generate_other_metrics(project)
    merged_df = pd.merge(und_metric_on_group_df, other_metric_on_group_df, on='clone_group_tuple', how='inner')
    merged_df.to_csv('../data/clones/%s_group_metric.csv' % project, index=False)


"""
Main function to execute the program.
"""
if __name__ == '__main__':
    if len(sys.argv) > 1:
        project = sys.argv[1]
    
    init_commit_modification_dict(project)

    # given clone_path and clone_range, retrieve clone_name
    clone_class_dict_4_clone = defaultdict(defaultdict)
    clone_class_dict_4_clone_path = '../data/clones/%s_clone_class_dict_4_clone.pkl' % project
    
    if os.path.exists(clone_class_dict_4_clone_path):
        with open(clone_class_dict_4_clone_path, 'rb') as handle:
            clone_class_dict_4_clone = pickle.load(handle)
    else:
        clone_class_dict_4_clone = get_clone_class(project)
        with open(clone_class_dict_4_clone_path, 'wb') as handle:
            pickle.dump(clone_class_dict_4_clone, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metric_on_group_df = pd.DataFrame()

    # loading genealogy file
    genealogy_df = pd.read_csv('../data/clones/%s_genealogies.csv' % project)

    # traverse the genealogy
    for commit_id in genealogy_df['start_commit'].unique():
        commit_metric_dict = get_metrics_by_commit(project, commit_id)  # metrics from understand tool
        commit_groups = genealogy_df[genealogy_df['start_commit'] == commit_id]['clone_group_tuple'].tolist()
        # Create a copy of the nested dictionary for the current commit_id
        clone_dict_commit_copy = clone_class_dict_4_clone[commit_id].copy()
        # Iterate over the second level keys (clone)
        for clone in clone_class_dict_4_clone[commit_id].keys():
            # Replace '/' with '.' and '.java:' with ':'
            new_clone = clone.replace('/', '.').replace('.java:', ':')
            # Update the nested dictionary with the modified key
            clone_dict_commit_copy[new_clone] = clone_class_dict_4_clone[commit_id][clone]
        # Update the original dictionary with the modified nested dictionary
        clone_class_dict_4_clone[commit_id] = clone_dict_commit_copy
        # traverse the groups in each commit
        for group in commit_groups:
            metric_on_group = Counter(["CountInput","CountLine","CountLineBlank","CountLineCode"
                 ,"CountLineCodeDecl","CountLineCodeExe"
                 ,"CountLineComment","CountOutput","CountPath","CountSemicolon"
                 ,"CountStmt","CountStmtDecl","CountStmtEmpty"
                 ,"CountStmtExe","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential"
                 ,"EssentialStrictModified","Knots","RatioCommentToCode","MaxEssentialKnots","MaxNesting"
                 ,"MinEssentialKnots","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential"
                ])
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
                parts = clone.split(':', 2)
                # Extract the parts before and after the second semicolon
                function_path = parts[0]
                functoin_start_end = parts[1]
                function_name = ':'.join(parts[2:])
                func_name = clone_class_dict_4_clone[commit_id][":".join([function_path, functoin_start_end])]
                clone_str = ".".join([clone_path, func_name]).strip()  # there might be spaces
                
                clone_metrics = search_clone(commit_metric_dict, clone_str)
                if clone_metrics is not None:
                    # only need the metrics on method level
                    clone_metrics = {key: val for key, val in clone_metrics.items() if
                                     key in ["CountInput","CountLine","CountLineBlank","CountLineCode"
                 ,"CountLineCodeDecl","CountLineCodeExe"
                 ,"CountLineComment","CountOutput","CountPath","CountSemicolon"
                 ,"CountStmt","CountStmtDecl","CountStmtEmpty"
                 ,"CountStmtExe","Cyclomatic","CyclomaticModified","CyclomaticStrict","Essential"
                 ,"EssentialStrictModified","Knots","RatioCommentToCode","MaxEssentialKnots","MaxNesting"
                 ,"MinEssentialKnots","SumCyclomatic","SumCyclomaticModified","SumCyclomaticStrict","SumEssential"
                ]}  
                    metric_on_group += Counter(clone_metrics)  # aggregate the metrics for clone group
                    # metric_on_group.update(Counter(clone_metrics))
            if clone_count:
                metric_on_group_dict = dict(metric_on_group)

                # get the average metric value
                metric_on_group_dict = {k: v / clone_count for k, v in metric_on_group_dict.items()}

                metric_on_group_dict.update({'clone_group_tuple': group})

                # metric_on_group_df = metric_on_group_df.append(metric_on_group_dict, ignore_index=True)
                # Convert metric_on_group_dict to a DataFrame
                new_data_df = pd.DataFrame.from_dict(metric_on_group_dict, orient='index').T
                # Concatenate the new data DataFrame with the original DataFrame
                metric_on_group_df = pd.concat([metric_on_group_df, new_data_df], ignore_index=True)
    combine_und_other_metrics(metric_on_group_df)

    print("done successfully !")
