import os, sys, time
import pandas as pd


"""
Main function to execute the program.
"""
if __name__ == '__main__':
    project = 'glide'
    if len(sys.argv) > 1:
        project = sys.argv[1]

    # read in metrics
    metric_on_group_df = pd.read_csv('../data/clones/%s_group_metric.csv' % project)
    for col_name in metric_on_group_df.columns:
        metric_on_group_df = metric_on_group_df.fillna({col_name: 0})
    time.sleep(5)

    # combine with label
    label_df = pd.read_csv('../data/clones/%s_label.csv' % (project))
    dataset = pd.merge(metric_on_group_df, label_df, how='inner', on='clone_group_tuple')#.drop_duplicates()
    dataset.to_csv('../data/clones/%s_raw_dataset.csv' % (project), index=False)
    time.sleep(5)

    print("done!")
