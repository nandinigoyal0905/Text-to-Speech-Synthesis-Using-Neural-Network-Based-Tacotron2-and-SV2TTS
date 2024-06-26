"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
# -*- coding:UTF-8 -*-
import numpy as np
import os
import tensorflow as tf
from emotion_encoder.Model import TIMNET_Model
import argparse
from emotion_encoder.utils import get_mfcc
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--test_path', type=str, default='saved_models/default/INTERSECT_46_dilation_8_dropout_05_add_esd_npairLoss')
parser.add_argument('--data', type=str, default='ESD_test')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--bidirection', type=bool, default=True)
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10
else:
    args.dilation_size = 8
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
session = tf.compat.v1.Session(config=config)
print(f"###gpus:{gpus}")

data = np.load("emotion_encoder/MFCC/"+args.data+".npy",allow_pickle=True).item()
x_source = data["x"]
y_source = np.argmax(data["y"], axis=1)

CLASS_LABELS_finetune = ("angry", "fear", "happy", "neutral","sad")
CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")#emovo
INTERSECT_CLASS_LABELS = ("angry", "happy", "neutral", "sad", "surprise")
ESD_CLASS_LABELS = ("angry", "happy", "neutral", "sad", "surprise")
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
               "EMODB": EMODB_CLASS_LABELS,
               "EMOVO": EMOVO_CLASS_LABELS,
               "IEMOCAP": IEMOCAP_CLASS_LABELS,
               "RAVDE": RAVDE_CLASS_LABELS,
               "SAVEE": SAVEE_CLASS_LABELS,
               "INTERSECT": INTERSECT_CLASS_LABELS}
CLASS_LABELS = CLASS_LABELS_dict["INTERSECT"]

model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
model.create_model()
x_feats = model.infer(x_source, model_dir=args.test_path)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import pandas as pd
import seaborn as sns
 
# We want to get UMAP embedding with 2 dimensions
reducer = umap.UMAP(int(np.ceil(np.sqrt(y_source.size))), metric="cosine")
umap_result = reducer.fit_transform(x_feats)
print(umap_result.shape)
# (1000, 2)
# Two dimensions for each of our images
 
# Plot the result of our UMAP with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
umap_result_df = pd.DataFrame({'x': umap_result[:,0], 'y': umap_result[:,1], 'label': y_source})
umap_result_df["label"]=umap_result_df["label"].apply(lambda x:ESD_CLASS_LABELS[x])
fig, ax = plt.subplots(1)
sns.scatterplot(x='x', y='y', hue='label', data=umap_result_df, ax=ax,s=40)
lim = (umap_result.min()-5, umap_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax.set_title('UMAP visualization of emotion speech test dataset')
if not os.path.exists("dim_reduction_results"):
    os.mkdir("dim_reduction_results")
plt.savefig("emotion_umap.png", dpi=500)
plt.savefig("dim_reduction_results/emotion_umap.png", dpi=500)
