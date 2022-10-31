# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:10:33 2022

@author: wsc72
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from pyvis.network import Network

""" import data. Only Demo data is provided. """
path = "YourPathHere"
os.chdir(path)
df = pd.read_csv("flv_notes_inputs_demo.csv")
df.columns = [["CAS#", "name", "notes"]]
df = df[["name", "notes"]].dropna()

""" Data cleaning for flavor notes col: df_f is the cleaned dataset"""
df["name"] = df["name"].apply(lambda x:  x.str.strip().str.lower()) 
df["notes"] = df["notes"].apply(lambda x: x.str.split(", | "))

stop_words = stopwords.words('english')
stop_words.extend(['slight', "nuances", "nuances.", "note", "notes", "nuance", "slightly", "undernotes", "back", "background", "de", "impact", "like", "mild", "strong"]) # no need for "stop_words = ", add it cause issue

def remove_stopwords(df_series):
    filtered = [i for i in df_series[0] if not i in stop_words]
    return filtered 

def lemmatize_text(df_series):
    lemmatizer = WordNetLemmatizer()
    lmed = [lemmatizer.lemmatize(i) for i in df_series[0]]
    return lmed

def replace_synonyms(df_series):  
    synonyms_dict = {"anisic": "anise", "bready": "bread", "caramellic": "caramel", "cheesy": "cheese",
                      "citric": "citrus", "cogac": "congac", "fat": "fatty", "fruitti": "fruity", "fruit": "fruity", "frutti": "fruity",
                      "jasmin": "jasmine", "menthol": "mentholic", "musk": "musky", "rummy": "rum",  "tobaccolike": "tobacco",
                      "winey": "wine", "wood": "woody"}  
    cleaned = [synonyms_dict[i] if i in synonyms_dict.keys() else i for i in df_series[0]]
    return cleaned

df["notes"] = df["notes"].apply(lambda x: remove_stopwords(x), axis =1)
df["notes"] = df["notes"].apply(lambda x: lemmatize_text(x), axis =1)
df["notes"] = df["notes"].apply(lambda x: replace_synonyms(x), axis =1)

# Convert notes to bag of words. 
df["notes_cvt"] = df.notes.applymap(lambda x: " ".join(x)) 
notes_lst = df.notes_cvt.values.tolist()
notes_lst = [i[0] for i in notes_lst]
vectorizer = CountVectorizer()
notes_matrix = vectorizer.fit_transform(notes_lst)
df_f = pd.DataFrame(notes_matrix.toarray(), columns=vectorizer.get_feature_names())

# Apply the following line ONLY when "like", "strong", "frutti" appear as Cols
#df_f.drop(["like", "strong", "frutti"], axis = 1, inplace = True)

#Remove notes appear twice for an ingred
df_f = df_f.replace(2, 1)
df_f["name"] = df["name"]

""" Exploratary analysis on cleaned dataset (df_f) """
# most frequent notes (>10), ingredients with most flv notes
#How to use diff color for the top 3/5??
df_sum = df_f[df_f.columns[:-1]].sum()
df_sum.sort_values(ascending =False, inplace=True)
df_plt = df_sum[df_sum.apply(lambda x: x>10)]
mask = df_plt >= 25
colors = np.array(['#1f77b4']*len(df_plt))
colors[mask.values] = 'g'
df_plt.plot(kind = "bar", title = "Most frequent flavor notes", rot = 45, color = colors)
plt.show()

df_sum2 =  df_f.sum(axis = 1)
df_sum2.index = df_f.name #set_index for df, pd.series has reset_index but original index is dropped or become abother col
df_sum2.sort_values(ascending=False, inplace=True)
df_plt2 = df_sum2[df_sum2 >= 9]
mask = df_plt2 >= 12
colors = np.array(['#1f77b4']*len(df_plt2))
colors[mask.values] = 'g'
ax = df_plt2.plot(kind = "barh", title = "Ingredients with most flavor notes", color = colors, figsize = (10, 5))
ax.invert_yaxis()
plt.show()

"""  Flavor mapping """

# chemical overlapping: co-occur prob
# co-occurance prob is normalized point-wise mutual info btw pairs of flv ingredients

def npmi(df_f, df_sum, term1, term2):
    freq_1 = df_sum[term1]/len(df_f)
    freq_2 = df_sum[term2]/len(df_f)
    df_temp = df_f[[term1, term2]].sum(axis = 1)
    freq_co = len(df_f[df_temp == 2])/len(df_f)
    if freq_co/(freq_1*freq_2) == 0: npmi = -1 # or 0??
    else:     
        pmi = math.log(freq_co/(freq_1*freq_2))  # log(0) --> -infinity
        npmi = pmi/(-math.log(freq_co))
    return npmi

#Remove notes that only appeared once
df_sum = df_sum[df_sum.apply(lambda x: x!=1)]
df_f_ing = df_f.T
df_f_ing.columns = df_f_ing.loc["name"]
df_f_ing = df_f_ing[0:-1]
df_f_ing = df_f_ing.loc[df_sum.index]

df_npmi_ing = pd.DataFrame(columns = ["Ing1", "Ing2", "npmi"]) 
ing1_lst, ing2_lst, npmi_lst = [[] for i in range(3)]   
ing_lst = list(df_sum2.index)
for i in range(len(ing_lst)):
    ing1 = ing_lst[i]
    for j in range(i+1, len(ing_lst)):
        ing2 = ing_lst[j]
        npmi_temp = npmi(df_f_ing, df_sum2, ing1, ing2) 
        #use df_sum here only consider notes appear >=2
        ing1_lst.append(ing1)
        ing2_lst.append(ing2)
        npmi_lst.append(npmi_temp)
df_npmi_ing.Ing1 = ing1_lst
df_npmi_ing.Ing2 = ing2_lst
df_npmi_ing.npmi = npmi_lst

df_npmi_ing.sort_values("npmi", ascending = False, inplace=True)

"""  Graph visualization for ingredient similarity 
Note: threshold of 0.55 was applied for the full dataset, which should be adjusted as needed.

"""
df_node = df_npmi_ing[df_npmi_ing["npmi"] >= 0.55]

# Initial exploration to identify clusters of ingredients
G = nx.from_pandas_edgelist(df_node, source = "Ing1", target = "Ing2", edge_attr = "npmi")
net = Network(height='750px', width='100%')
net.from_nx(G)
net.show("flvmapping.html") 


# A total of 6 clusters were visually indetified and then manaully added in "ing_npmi_group.csv"
# The dataset of "ing_npmi_group.csv" can be directly imported for Graph visulization
df_node = pd.read_csv("ing_npmi_group.csv")
sources = df_node['Ing1']
targets = df_node['Ing2']
weights = df_node['npmi']
groups = df_node['Groups']
edge_data = zip(sources, targets, weights, groups)
got_net = Network(height='750px', width='100%')

for src, dst, w, g in edge_data:
    if g == "G1":
        got_net.add_node(src, src, title=src, size  =10, color = "royalblue")
        got_net.add_node(dst, dst, title=dst, size  =10, color = "royalblue")
        #got_net.add_edge(src, dst, value=w)
    if g == "G2":
        got_net.add_node(src, src, title=src, size  =10, color = "palevioletred")
        got_net.add_node(dst, dst, title=dst, size  =10, color = "palevioletred")
    if g == "G3":
        got_net.add_node(src, src, title=src, size  =10, color = "darkgoldenrod")
        got_net.add_node(dst, dst, title=dst, size = 10, color = "darkgoldenrod")   
    if g == "G4":
        got_net.add_node(src, src, title=src, size=10, color = "olivedrab")
        got_net.add_node(dst, dst, title=dst, size=10, color = "olivedrab")
    if g == "G5":
        got_net.add_node(src, src, title=src, size=10, color = "darksalmon")
        got_net.add_node(dst, dst, title=dst, size=10, color = "darksalmon")
    if g == "G6":
        got_net.add_node(src, src, title=src, size=10, color = "seagreen")
        got_net.add_node(dst, dst, title=dst, size=10, color = "seagreen")
    got_net.add_edge(src, dst, value=w)

got_net.show("grouped_flv_map.html")


