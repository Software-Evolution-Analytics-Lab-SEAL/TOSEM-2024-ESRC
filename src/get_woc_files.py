
import pickle, os, sys, gzip, subprocess, shutil, itertools, paramiko
from itertools import islice
sys.path.append("..")
sys.path.append(".")
from time import time
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
# from oscar import Commit, File, GitObject, Blob, Project, Tree
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from functools import reduce
from pyspark.sql import DataFrame
import gzip
from itertools import islice
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import SparkSession
from concurrent.futures import ProcessPoolExecutor
import pickle, os, sys, gzip, subprocess, shutil, itertools, re
from itertools import islice
sys.path.append("..")
sys.path.append(".")
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
# from oscar import Commit, File, GitObject, Blob, Project, Tree
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from functools import reduce
from pyspark.sql import DataFrame
import gzip
from itertools import islice
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import pickle, os, sys, gzip, subprocess, shutil, itertools, re
from itertools import islice
sys.path.append("..")
sys.path.append(".")
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
# from oscar import Commit, File, GitObject, Blob, Project, Tree
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from functools import reduce
from pyspark.sql import DataFrame
import gzip
from itertools import islice
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import functions as F
from pyspark.sql.functions import concat_ws
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import collect_set, size, count


"""
    Functionality: Establishes an SSH connection to a remote server and uses SFTP to download specific files. The function is 
    configured to handle SSH key policies automatically and safely close connections upon completing the file transfers. It 
    downloads a predetermined set of files based on an index range, saving them to a specified local directory.

    Expected Input:
    None directly provided to the function. The function uses hard-coded local and remote path structures with an index to
    identify specific files. This index-based approach assumes a predictable naming convention and directory structure for 
    both source and destination paths.

    Expected Output:
    None returned to the caller. The primary output is the files that are downloaded and saved to the local filesystem.

    Side Effects:
    - Uses the network to download files, which can consume significant bandwidth and may be impacted by network latency or 
      stability.
    - Interacts with the file system to save the downloaded files, which requires adequate permissions and sufficient disk space.
    - Establishes an SSH connection and modifies the SSH key policy, which could have security implications if not properly 
      managed in a broader application context.
    - The function assumes the presence of specific files on the remote server, and will fail if these files are not available 
      or the paths are incorrect.
    - The function does not handle potential exceptions from network issues or file system errors, which might result in 
      incomplete file transfers or application crashes if not managed.

    Notes:
    - Ensure that the remote server's host key is known or appropriately handled to avoid security issues.
    - Consider adding error handling to manage exceptions during connection setup, file transfer, and file operations to improve 
      robustness.
    - Review and potentially externalize the file indices and paths to configuration settings to make the function more flexible 
      and maintainable.
"""
def download_files():

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = ssh.open_sftp()

    for i in range(2):
        local_path = f"../data/c2PtAbflPkgFullU/c2PtAbflPkgFullU{i}.s"
        remote_path = f"/da3_data/basemaps/gz/c2PtAbflPkgFullU{i}.s"
        sftp.get(remote_path, local_path)

    sftp.close()
    ssh.close()



"""
    Functionality: Processes CSV files by loading them in chunks, filtering for entries that pertain to Python code ('PY' language),
    and then saving these filtered chunks to Parquet files for efficient storage and retrieval. The function handles file reading,
    filtering based on language, and writes to Parquet in a directory structure. It uses the set of non-problematic indices to 
    determine which files to process, skipping those known to cause issues.

    Expected Input:
    None directly provided to the function. It internally uses predefined parameters such as the list of column names, 
    the chunk size for reading CSV files, and a base directory for output files. It assumes that a list of problematic indices
    (`indices_problematic`) is defined outside the function, which it uses to exclude certain files from processing.

    Expected Output:
    None returned to the caller. The function's primary output is the set of Parquet files written to disk, each corresponding
    to a chunk of the original CSV files that contains only Python language data.

    Side Effects:
    - Creates new directories if they do not exist using `os.makedirs`, affecting the file system.
    - Reads large CSV files in chunks, which could consume significant memory and CPU resources during execution.
    - Writes multiple Parquet files to the disk, involving I/O operations that may impact system performance.
    - The function includes error handling that silently ignores exceptions, which could suppress important errors during 
      file reading and writing.
    - Uses a break statement after processing the first problematic index, which seems intended for testing or debugging 
      but might be an error if left in production code (this should be clarified or removed if not needed).

    Notes:
    - Ensure that the environment where this function runs has sufficient disk space and memory to handle large-scale data processing.
    - It's important to review and potentially revise the exception handling to provide more visibility into any issues that occur.
    - The specific handling of indices and the decision to break after the first loop iteration should be re-evaluated for accuracy
      and completeness in a production setting.
"""
def load_files():
    chunksize = 1e6
    cols = ['commit', 'repo', 'timestamp', 'author', 'blobhash', 'filename', 'language']# , 'techdeps']
    parquet_dir_py = "/data/woc_data/files_python_test"
    os.makedirs(parquet_dir_py, exist_ok=True)
    
    df_chunk_py_list = []
    indices = set(list(range(128))) - set(indices_problematic)
    #for i in range(128):
    for i in indices_problematic:
        partition_filename = f'../data/c2PtAbflPkgFullU/c2PtAbflPkgFullU{i}.s'
        #print(partition_filename)
        file_size = os.path.getsize(partition_filename)
        # print(file_size)
        with gzip.open(partition_filename, 'rt', errors='ignore') as fp:
            # for line in itertools.islice(fp, 1):
            #     print(line)
    
            chunk_count = 0
            try:
                for df_chunk in pd.read_csv(fp, chunksize=chunksize, sep=';', usecols=range(7), header=None, names=cols):
                #for df_chunk in pd.read_csv(fp, chunksize=chunksize, sep=';', usecols=range(7), header=None, names=cols, engine='python', quoting=csv.QUOTE_NONE):
                    df_chunk_py = df_chunk[df_chunk['language'] =='PY']
    #   

                    df_chunk_py_path = os.path.join(parquet_dir_py, f"c2PtAbflPkgFullU{i}_chunk{chunk_count}_py.parquet")
                    df_chunk_py.to_parquet(df_chunk_py_path)
    #   
                    chunk_count += 1
            except Exception as e:
                pass
        break


"""
    Functionality: Reads a specified CSV file containing software repository data, filters it by programming language, and writes 
    the filtered data back to a new CSV file. The function is designed to preprocess data specific to Python files. It initializes a 
    Spark session, defines a schema for the CSV data, and performs filtering based on the 'language' column.

    Expected Input:
    i (int): An index representing a specific file to be processed. This index is used to generate the filename dynamically, ensuring 
    the correct file is read and processed.

    Expected Output:
    None directly returned to the caller. The function instead writes a new CSV file containing only the entries where the 'language' 
    column is 'PY'. This file is saved to a specified path with an index appended to its name to denote its specific segment.

    Side Effects:
    - Reads a CSV file from disk based on a dynamically constructed filename that includes the input index.
    - Creates a new CSV file on disk containing filtered data, which involves I/O operations.
    - Outputs to standard output via the `show` method, which prints the DataFrame contents to the console.
    - Utilizes system resources to run Spark tasks, which could affect overall system performance depending on the size of the data 
      and available system resources.
"""
def preprocess_file(i):
    print("i = ", i)
    # Write the list to a CSV file
    spark = SparkSession.builder.appName("CSVReader").getOrCreate()
    schema = StructType() \
        .add("commit", StringType(), True) \
        .add("repo", StringType(), True) \
        .add("timestamp", StringType(), True) \
        .add("author", StringType(), True) \
        .add("blobhash", StringType(), True) \
        .add("filename", StringType(), True) \
        .add("language", StringType(), True)
    df = spark.read.csv("../data/tmp/c2PtAbflPkgFullU%s_processed.csv"% str(i), sep=";", schema=schema, header=False)

    # save python
    df_py = df.filter(df.language == 'PY')
    df_py.write.csv("../data/file_partitions_py/c2PtAbflPkgFullU{%s}_py.csv" % str(i), sep=";", header=True, mode="overwrite")      
    
    df_py.show()



"""
    Functionality: Processes multiple CSV files to extract, combine, and analyze data related to code clones across software repositories. 
    The function sets up a Spark session with specific memory configurations, reads CSV files into DataFrames, and merges them.
    It then constructs URLs for each clone based on repository metadata, identifies and filters clones based on duplication across projects,
    and finally outputs results including a list of URLs and associated metadata for further analysis.

    Expected Input:
    No direct input parameters are passed to the function, but it is expected to operate within an environment where the necessary CSV 
    files are available under a predefined directory structure. It also requires a Spark environment properly configured for the 
    available system resources.

    Expected Output:
    None directly from the function, but several side effects include:
    - Reading and processing CSV files from a specified path.
    - Creating and modifying Spark DataFrames to consolidate clone data.
    - Writing results to stdout during the process and to CSV files at the conclusion.
    - Potentially exporting lists of clone identifiers for further processing or review.

    Side Effects:
    - Files are read and written to disk.
    - Spark SQL temporary views or data may persist in memory after execution.
    - External side effects include the usage of system I/O and significant memory and CPU resources.
"""
def get_clones():
    spark = SparkSession.builder.appName("ReadCSV") \
        .config("spark.executor.memory", "64g") \
        .config("spark.driver.memory", "64g") \
        .config("spark.memory.offHeap.size", "64g") \
        .config("spark.driver.maxResultSize", "64g") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # Assuming dfs is a list containing your 128 DataFrames
    df_list = []
    max_index = 0
    for i in tqdm(range(2)): # Now, indexed_dfs contains your 128 DataFrames with continuous indices across them.
        # Read the CSV file
        df_partition = spark.read.csv("../data/file_partitions_py/c2PtAbflPkgFullU%s_py.csv"%str(i), sep=";", header=True)
    
        # df_partition = df_partition.withColumn("index", monotonically_increasing_id() + max_index)
        # max_index = df_partition.agg({"index": "max"}).collect()[0][0] + 1
        df_list.append(df_partition)
    
    combined_df = df_list[0]
    for df in df_list[1:]:
        combined_df = combined_df.union(df)
    combined_df.show() # 
    combined_df.persist()

    combined_df = combined_df.withColumn(
        "URL",
        F.concat(
            F.lit("https://github.com/"),
            F.expr("regexp_replace(repo, '^(.*?)_', '$1/')"), 
            F.lit("/blob/"),
            F.col("commit"),
            F.lit("/"),
            F.col("filename")
        )
    )

    df_with_index_combined = df_with_index.withColumn("filepath", concat_ws("|", "repo_index", "commit", "filename_index"))
    df_with_index_combined_selected = df_with_index_combined.select("blobhash", "filepath", "timestamp", "repo_index")
    df_with_index_combined_selected_ordered = df_with_index_combined_selected.orderBy("blobhash", "timestamp")
    df_with_index_combined_selected_ordered = df_with_index_combined_selected_ordered.repartition("blobhash")
    
    # Find replicates
    replicates = df_with_index_combined_selected_ordered.groupBy("blobhash").count().filter("count > 1")
    replicates_list = replicates.collect()
    #   Extract the 'blobhash' values
    blobhash_list = [row['blobhash'] for row in replicates_list]
    with open('blobhash_list_py.pkl', 'wb') as fp:
        pickle.dump(blobhash_list, fp)

    spark_filter = SparkSession.builder.appName("filter").getOrCreate()
    blobhash_df = spark_filter.createDataFrame([(bh,) for bh in blobhash_list], ["blobhash"])
    df_clones = df_with_index_combined_selected_ordered.join(blobhash_df, "blobhash") # Broadcast join: 20min
     
    # Show the results
    df_clones.show()
    df_clones.persist()
    
    # Convert the list to a DataFrame
    blobhash_df = spark.createDataFrame([(bh,) for bh in blobhash_cloned_within_list], ['blobhash'])
    
    # Broadcast the DataFrame
    broadcast_blobhash_df = broadcast(blobhash_df)
    
    # Join the DataFrames and count the number of records
    num_records = df_clones.join(broadcast_blobhash_df, 'blobhash').count()
    print(num_records)

    combined_df_grouped = combined_df.groupBy("blobhash").agg(
        F.collect_list("URL").alias("URL_list"),
        F.collect_list("timestamp").alias("timestamps"),
        F.collect_set("repo").alias("repo_set"),
    )

    # Order the URL list by timestamps
    combined_df_grouped_ordered = combined_df_grouped.withColumn(
        "URL_list_ordered",
        F.expr("sort_array(arrays_zip(timestamps, URL_list), true).URL_list")
        ).drop("URL_list", "timestamps")
    
    df_clones = df_with_index_combined_selected_ordered.filter(col("blobhash").isin(blobhash_list))
    # Show the results
    df_clones.show()

    # Group by blobhash and aggregate
    df_clones_aggregated = df_clones.groupBy("blobhash").agg(
        collect_set("repo_index").alias("distinct_repo_indices"),
    )
    
    # Filter to get replicated blobhash with only one distinct repo_index
    df_clones_within = df_clones_aggregated.filter((size("distinct_repo_indices") == 1))
    
    # Show the results
    df_clones_within.show()

    # Convert the list to a DataFrame
    df_blobhash_cloned_cross = spark.createDataFrame([(t,) for t in blobhash_cloned_cross_list], ['blobhash'])
    
    # Broadcast the small DataFrame and join
    df_clones_cross = df_clones.join(broadcast(df_blobhash_cloned_cross), 'blobhash')
    
    df_clones_cross.show()
    df_clones_cross.coalesce(1).write.csv('df_clones_cross.csv', header=True, mode='overwrite')

    # Filter to get blobhashes with only one URL in the URL list
    


if __name__ == "__main__":
    print("hello")
    start_time = time()
    # download_files()

    preprocess_file(76)
#
    total_time = time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")