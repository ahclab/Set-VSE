#!/usr/bin/env python

import argparse
import os
import sys
import random

from PIL import Image
from tqdm import tqdm
import torch
import clip
import numpy as np
from copy import deepcopy
import matplotlib
from matplotlib import pyplot as plt
# import open_clip
import clip

from predict_clip_ret_score import load_clip, id2vocab, tokenizer, clip_pred, Recall_at_K

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="ViT-B/32",choices=["ViT-B/32", "RN50"])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="results_setVSE")
parser.add_argument("--dataset_type", type=str, default="test", choices=["train", "valid", "test"])
parser.add_argument("--img_type", type=str, default="global", choices=["global", "partial", "hybrid"])
parser.add_argument("--text_type", type=str, default="global", choices=["global", "partial", "hybrid"])
parser.add_argument("--IPOT", action="store_true", help="Use IPOT for VSE")

args = parser.parse_args()

# Load CLIP model
device = args.device
model, img_preprocess = load_clip(args.model_name, device=device)

# model, _, img_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load dataset
from densely_captioned_images.dataset.impl import get_summarized_dataset_with_settings, DenseCaptionedDataset
def load_dataset(dataset_type):
    ds: DenseCaptionedDataset = get_summarized_dataset_with_settings(
        split=dataset_type, 
        load_base_image=True, 
        load_subcaptions=True,
        negative_source="swaps",
        negative_strategy="rand",
        count=1e30)
    return ds
ds = load_dataset(args.dataset_type)
if args.dataset_type == "train":
    random.seed(args.seed)
    random.shuffle(ds)

# Load image database
@torch.no_grad()
def load_image_db(image_db_dir, ds, model, img_preprocess, device, img_type):
    if os.path.exists(image_db_dir):
        print("Loading image db from", image_db_dir)
        satate_dict = torch.load(image_db_dir)
        image_db = satate_dict["image_db"]
        img_group_ids = satate_dict["img_group_ids"]
    else:
        # create image db
        image_db = []
        img_group_ids = []
        with torch.no_grad():
            for i, sample in enumerate(ds):
                global_set = sample[0] # 画像と説明文のセット
                partial_sets = sample[1:] # 部分画像と説明文のセット（複数）
                if img_type == "global" or img_type == "hybrid" or len(partial_sets) == 0:
                    g_img = global_set["image"]
                    img = img_preprocess(Image.fromarray(g_img))
                    img = img.unsqueeze(0).to(device)
                    img_feat = model.encode_image(img)
                    image_db.append(img_feat.to("cpu"))
                    img_group_ids.append(i)
                if (img_type == "partial" or img_type == "hybrid") and len(partial_sets) > 0:
                    for partial_set in partial_sets:
                        p_img = partial_set["image"]
                        img = img_preprocess(Image.fromarray(p_img))
                        img = img.unsqueeze(0).to(device)
                        img_feat = model.encode_image(img)
                        image_db.append(img_feat.to("cpu"))
                        img_group_ids.append(i)

            image_db = torch.cat(image_db, dim=0)
            img_group_ids = torch.tensor(img_group_ids)

        state_dict = {
            "image_db": image_db,
            "img_group_ids": img_group_ids
        }
        torch.save(state_dict, image_db_dir)
    print("image_db.shape", image_db.shape)
    return image_db, img_group_ids

image_db_dir = args.dataset_type + "_img_db_"+ args.img_type
image_db, img_group_ids = load_image_db(image_db_dir, ds, model, img_preprocess, device, args.img_type)

@torch.no_grad()
def load_text_db(text_db_dir, ds, model, device, text_type):
    if os.path.exists(text_db_dir):
        print("Loading text db from", text_db_dir)
        state_dict = torch.load(text_db_dir)
        text_db = state_dict["text_db"]
        text_group_ids = state_dict["text_group_ids"]
    else:
        # create text db
        text_db = []
        text_group_ids = []
        with torch.no_grad():
            for i, sample in enumerate(ds):  
                global_set = sample[0] # 画像と説明文のセット
                partial_sets = sample[1:] # 部分画像と説明文のセット（複数）
                if text_type == "global" or text_type == "hybrid" or len(partial_sets) == 0:
                    g_caption = global_set["caption"]
                    texts = clip.tokenize(g_caption, truncate=True).to(device)
                    text_feat = model.encode_text(texts)
                    text_db.append(text_feat.to("cpu"))
                    text_group_ids.append(i)
                if (text_type == "partial" or text_type == "hybrid") and len(partial_sets) > 0:
                    for partial_set in partial_sets:
                        p_caption = partial_set["caption"]
                        texts = clip.tokenize(p_caption, truncate=True).to(device)
                        text_feat = model.encode_text(texts)
                        text_db.append(text_feat.to("cpu"))
                        text_group_ids.append(i)

            text_db = torch.cat(text_db, dim=0)
            text_group_ids = torch.tensor(text_group_ids)
        state_dict = {
            "text_db": text_db,
            "text_group_ids": text_group_ids
        }
        torch.save(state_dict, text_db_dir)
    print("text_db.shape", text_db.shape)
    return text_db, text_group_ids

text_db_dir = args.dataset_type + "_text_db_" + args.text_type
text_db, text_group_ids = load_text_db(text_db_dir, ds, model, device, args.text_type)

# restrict the size of the database
N_limit = 1000

# idsの中身の最大値はN_limitまでになるようにする
with torch.no_grad():
    mask = img_group_ids < N_limit
    img_group_ids = img_group_ids[mask]
    image_db = image_db[mask]

    t_mask = text_group_ids < N_limit
    text_group_ids = text_group_ids[t_mask]
    text_db = text_db[t_mask]

    image_db = image_db.float()
    text_db = text_db.float()

@torch.no_grad()
def IPOT(query, db, n_iter=50, beta=0.5, save_dir=None, query_group_ids=None, db_group_ids=None):

    if save_dir is not None:
        
        matplotlib.rc('xtick', labelsize=10) 
        matplotlib.rc('ytick', labelsize=10) 
        
    if query_group_ids is not None and db_group_ids is not None:
        src = [str(t) for t in query_group_ids]
        tgt = [str(t) for t in db_group_ids]

    embs1 = query.float()
    embs2 = db.float()

    n = len(embs1)
    m = len(embs2)

    m_one = torch.ones(m,1)
    n_one = torch.ones(n,1)

    sigma = m_one/m
    T = torch.matmul(n_one, m_one.T)

    norm1 = torch.norm(embs1, dim=1)
    norm2 = torch.norm(embs2, dim=1)

    cossim = 1 - torch.matmul(embs1, embs2.T) / torch.matmul(norm1.unsqueeze(1), norm2.unsqueeze(0))

    #heatmap ref: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    A = torch.exp(-cossim/beta)

    count = 0
    for k in range(n_iter):
        Q = A * T
        delta = 1/n / torch.matmul(Q, sigma)
        sigma = 1/m / torch.matmul(Q.T, delta)
        
        T = torch.matmul(torch.diag(delta.squeeze()), Q)
        T = torch.matmul(T, torch.diag(sigma.squeeze()))
        T_t = T.T

        if (k+1) % 10 == 0 or k==0:
            if save_dir is not None:
                plt.clf()
                fig, ax = plt.subplots(figsize=(50,50))

                if img_group_ids is not None and query_group_ids is not None:
                    # We want to show all ticks...
                    ax.set_xticks(np.arange(len(src)))
                    ax.set_yticks(np.arange(len(tgt)))
                    # ... and label them with the respective list entries
                    ax.set_xticklabels(src)
                    ax.set_yticklabels(tgt)

                    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    #     rotation_mode="anchor")

                    # for i in range(len(src)):
                    #     for j in range(len(tgt)):
                    #         text = ax.text(i, j, "%1.2f" % T[i, j],
                    #             ha="center", va="center", color="k")
                    # fig.tight_layout()
                    
                ax.set_title("iter = {}".format(k+1), fontsize=10)
                im = ax.imshow(T.detach().numpy())

                savepath = save_dir + "/OT_alignment_"+str(count).zfill(2)+".png"
                print("figure saved to: ", savepath)
                plt.savefig(savepath)
            plt.close()
            count += 1 
            # print(T)
    return T # (N, M)

@torch.no_grad()
def pooling_sim(sim, query_group_ids, db_group_ids, pooling="average"):
    array = sim.detach().numpy()
    query_group_ids = query_group_ids.detach().numpy()
    db_group_ids = db_group_ids.detach().numpy()

    row_sums = np.array([array[query_group_ids == label].sum(axis=0) for label in np.unique(query_group_ids)])

    column_sums = np.array([row_sums[:, db_group_ids == label].sum(axis=1) for label in np.unique(db_group_ids)])

    new_sim = torch.tensor(column_sums.T)
    return new_sim

# Recall_at_K
@torch.no_grad()
def recall_at_k(sim):
    """
    VSE:
    sim = sim # (N: image, B: query text)

    IPOT: 
    sim = T.T # (N: image, B: query text)
    """

    Ks = [1,5,10,100]
    n_count = {K:0 for K in Ks}
    n_total = 0
    max_K = max(Ks)

    ordered_index = torch.argsort(sim, dim=0, descending=True) # (N,B)

    query_ids = torch.arange(sim.shape[1])
    db_ids = torch.arange(sim.shape[0])

    # Extract the index of query_id that matches db_id 
    ordered_index = ordered_index.cpu()
    query_ids = query_ids
    ordered_db_ids = db_ids[ordered_index].cpu()
    condition = (ordered_db_ids == query_ids)
    assert torch.any(condition, dim=0).all() == True, "No query_id matches db_id"
    query_id_ranks = torch.argmax(condition*1.0, dim=0) # (B)

    # Calculate matching rate in image_id and query_id on each Top-K
    for k in n_count.keys():
        n_count[k] += torch.sum(query_id_ranks < k).item()
    n_total += query_ids.shape[0]

    # image_ids that are ranked higher than query_id
    higher_ranked_image_ids = {}
    for query_id, query_id_rank in zip(query_ids, query_id_ranks):
        higher_ranked_image_ids[query_id.item()] = ordered_db_ids[:query_id_rank].numpy().tolist()

    recall = {K: n_count[K]/n_total for K in n_count.keys()}
    print(recall)

# Standard VSE
emb_text = text_db / torch.norm(text_db, dim=1).unsqueeze(1)
emb_img = image_db / torch.norm(image_db, dim=1).unsqueeze(1)
Text2Img_sim_partial = torch.matmul(emb_text, emb_img.T) 
Img2Text_sim_partial = torch.matmul(emb_img, emb_text.T) 

Text2Img_sim = pooling_sim(Text2Img_sim_partial, text_group_ids, img_group_ids)
Img2Text_sim = pooling_sim(Img2Text_sim_partial, img_group_ids, text_group_ids)
print("Standard VSE + pooling shape")
print(Text2Img_sim.shape)
print(Img2Text_sim.shape)

recall_at_k(Text2Img_sim)
recall_at_k(Img2Text_sim)

# Standard VSE + IPOT
if args.IPOT:
    save_dir_T2I = args.save_dir + "/IPOT_T2I"
    if not os.path.exists(save_dir_T2I):
        os.makedirs(save_dir_T2I)
    save_dir_I2T = args.save_dir + "/IPOT_I2T"
    if not os.path.exists(save_dir_I2T):
        os.makedirs(save_dir_I2T)

    Text2Img_OT_matrix = IPOT(
        query=text_db, 
        db=image_db,
        save_dir=save_dir_T2I, 
        query_group_ids=None, 
        db_group_ids=None)

    Img2Text_OT_matrix = IPOT(
        query=image_db, 
        db=text_db, 
        save_dir=save_dir_I2T, 
        query_group_ids=None, 
        db_group_ids=None)

state_dict = {
    "query_group_ids": text_group_ids,
    "db_group_ids": img_group_ids,
    "T": Text2Img_OT_matrix
}
torch.save(state_dict, save_dir_T2I + "/IPOT_alignment.pt")

state_dict = {
    "query_group_ids": img_group_ids,
    "db_group_ids": text_group_ids,
    "T": Img2Text_OT_matrix
}
torch.save(state_dict, save_dir_I2T + "/IPOT_alignment.pt")

Text2Img_OT_sim = pooling_sim(Text2Img_OT_matrix, text_group_ids, img_group_ids)
Img2Text_OT_sim = pooling_sim(Img2Text_OT_matrix, img_group_ids, text_group_ids)
print("Standard VSE + IPOT + pooling shape")
print(Text2Img_OT_sim.shape)
print(Img2Text_OT_sim.shape)

recall_at_k(Text2Img_OT_sim)
recall_at_k(Img2Text_OT_sim)

