# An Empirical Study on the Characteristics of Reusable Code Clones


### Disclaimer
The following project is shared for educational purposes only. 
The author and its affiliated institution are not responsible in any manner whatsoever for any damages, 
including any direct, indirect, special, incidental, 
or consequential damages of any character arising as a result of the use or inability to use this software.

The current implementation is not, in any way, intended, nor able to generate code in a real-world context. 
We could not emphasize enough that this project is experimental and shared for educational purposes only. 
Both the source code and the datasets are provided to foster future research in software engineering and are not designed for end users.

### Key components

**SRC**: The src/ directory contains all the scripts and implementations of the experiments in the paper. Each process is organized into separate files for clarity and modularity.

- 0_clone_detection.py contains the scripts to mine software repositories and conduct clone detection.   
- 1_purify_clone_detection_paratypes.py: After detecting clone classes from each of the commits from a Git repository, this script will extract clone classes into a JSON file from the raw clone results.   
- 2_build_genealogy.py: extract clone classes from the results of clone classes, then built the clone genealogy for each clone class.   
- 3_extract_metrics.py: extract basic metrics from the results of clone genalogies.
- 4_combine_metrics.py: combine the metrics from Understand tool and calculated metrics into a whole.
- 5_extract_quality_labels.py: extract quality metrics of clone genealogies.   
- 6_combine_metrics_label.py: map features to the corresponding label.   
- 7_model_training_parallel.py: use the tree classifiers and train the 60 project in parallel.   


The dataset related to this replication package can be downloaded from:
https://drive.google.com/file/d/1huPlVaoDQ9ks6HNH9UXIH5zCSi0vPQ70/view?usp=sharing

**Data**: The data/ directory includes the retired szz analysis result, just keep it in case it will be used in the future.

The project-url.xlsx includes all the projects including its links to github repository. 

Before running the clone detection, please make sure the subject system repository is cloned to the data/subject_systems. Use the command git clone command to do it.

### Set up

#### Prerequisites
- Python 2.7 or newer   
- NiCad clone detection tool: NiCad==6.2
  https://www.txl.ca/txl-nicaddownload.html 
- Understand code analysis tool build 1102: Understand==1.0.3 https://marketplace.visualstudio.com/items?itemName=scitools.understand
- urllib3==1.26.5

#### Install dependencies
pip install -r  requirements.txt

#### Create a new virtual environment depends on developer's preference

The second argument is the location to create the virtual environment. Generally, you can just create this in your project and call it .venv.
venv will create a virtual Python installation in the .venv folder.   
```python3 -m venv .venv```

Activate a virtual environment.   
```source .venv/bin/activate```

Deactivate a virtual environment.   
```deactivate```

### Usage
Prepare the data:

enter into the src folder where the scripts exists.   
```cd src```

get clone detection result for the project (i.e., glide) based on nicad   
```python 0_clone_detection.py glide nicad```

purify the clone result of the project (i.e., glide)   
```python 1_purify_clone_detection_paratypes.py glide```

get clone detection result for the project (i.e., glide) based on nicad   
```python 2_build_genealogy.py glide nicad```

get metrics of the project (i.e., glide)   
```python 3_extract_metrics.py glide```

combine all the metrics together of the project (i.e., glide)   
```python 4_combine_metrics.py glide```

extract the quality labels of the project (i.e., glide)   
```python 5_extract_quality_labels.py glide```

combine metrics with label of the project (i.e., glide)   
```python 6_combine_metrics_label.py glide```

#### Fine-tune the model:   
```python 7_model_training_parallel.py```


### FAQ
#### Will this project supports other languages?
Yes, this research project and will stay in the state described in the paper for consistency reasons. You are of course more than welcome to fork the repo and experiment yourself with other target platforms/languages.

#### How long does it take to run clone detection?
For large projects, it takes two weeks without stopping. For small projects, it takes 3-5 days to finish. For middle-sized projects, it may take 1 to 2 weeks to finish clone detection step.

#### How long does it take to train the model?
On GPU, it takes a little less than 5 hours for one dataset; 
so expect around (5 hours) \* (60 projects ) \* (5 classifiers) if you want to train the model.

