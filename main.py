#!/usr/bin/env python3

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sys
import csv

from cluster import Clustering, Topic_identifying
clustering_data = Clustering()
identifying_topic = Topic_identifying()

# IMPORT DATA
def main():
    df_raw = pd.read_csv("data/Final Dataset Fix.csv")
    get_department_div(df_raw)
    # fac = ['FKIP', 'FMIPA', 'FEB', 'FT', 'FP', 'FISIP', 'FK', 'FH']
    # process_division(df_raw, fac)

# CLUSTERING DATA AND IDENTIFY TOPIC FOR ALL DATA
def process_all(df_raw):
    clusters, k, ss, sc, sd, df_cluster = clustering_data.modeling_cluster(df_raw)
    list_topic = identifying_topic.identifying(df_cluster, k)

# CLUSTERING DATA AND IDENTIFY TOPIC PER DIVISIONS
def process_division(df_raw, data_list):
    # data divisons setting
    df_raw = df_raw.dropna(subset=['divisions'])
    total_data = len(df_raw)
    
    # csv setting
    csv_file = open('result/cluster_information_per_division.csv', 'w', newline="", encoding='utf-8') 
    writer = csv.writer(csv_file, sys.stdout, lineterminator='\n')
    writer.writerow(["faculty", "faculty_unique_relation", "total_data", "percentage", "cluster_number", "silhoutte_score", "calonski_harabasz_score", "davies_bouldin_score", "list_topic"])
    
    # create new dataframe for data csv
    dict_new = {}

    # run data per division
    for f in data_list:
        print(f)
        # process the clustering and identifying topic per divisions
        df = df_raw[df_raw["divisions"].str.contains(f)]
        x_vector, clusters, k, ss, sc, sd, df_cluster = clustering_data.modeling_cluster(df)
        list_topic = identifying_topic.identifying(df_cluster, k)
        clustering_data.visualizing_cluster(x_vector, clusters, k, f)
        # insert to dictionary
        dict_new["division"] = f
        dict_new["division_unique_relation"] = str(df["divisions"].unique().tolist()).replace('"', '')
        dict_new["total_data"] = int(len(df))
        dict_new["percentage"] = round(100*(len(df)/total_data), 2)
        dict_new["cluster_number"] = int(k)
        dict_new["silhoutte_score"] = float(ss)
        dict_new["calonski_harabasz_score"] = float(sc)
        dict_new["davies_bouldin_score"] = float(sd)
        dict_new["list_topic"] = list_topic
        # export to csv
        writer.writerow(dict_new.values())

# CHECK DATA INFORMATION
def check_info(df):
    # check how many data including the null values
    print("jumlah data", len(df))
    print("jumlah missing data per division", df["divisions"].isnull().sum())
    print("persentase missing data per division", 100*(df["divisions"].isnull().sum()/len(df)))
    print(df.isnull().sum())
    print(df.info())
    print(df.head())

def get_department_div(df):
    df = df.dropna(subset=['divisions'])
    print(df["divisions"])
    list_div = df["divisions"].tolist()
    list_div = list(set(list_div))
    list_div = str(list_div).replace('"', '').replace('[', '').replace(']', '').replace(' ', '').replace("'", "")
    print(list_div)
    print(type(list_div))
    li = list(list_div.split(","))
    print(sorted(li))
    print(type(li))
    print(len(li))
    li_unique = list(set(li))
    li_unique = sorted(li_unique)
    print(li_unique)
    print(type(li_unique))
    print(len(li_unique))
    # belum di crosscheck lagi dengan data di lppm

if __name__ == '__main__':
    main()