#!/usr/bin/env python

import argparse
import os
import sys

from PIL import Image
from tqdm import tqdm
import torch
import clip

from predict_clip_ret_score import load_clip, id2vocab, tokenizer, clip_pred, Recall_at_K

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="ViT-B/32",choices=["ViT-B/32", "RN50"])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="results_VSE")
parser.add_argument("--dataset_type", type=str, default="test", choices=["train", "valid", "test"])
parser.add_argument("--IPOT", action="store_true", help="Use IPOT for VSE")

args = parser.parse_args()

# Load CLIP model
device = args.device
model, img_preprocess = load_clip(args.model_name, device=device)

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

# Load image database
def load_image_db(image_db_dir, ds, model, img_preprocess, device):
    if os.path.exists(image_db_dir):
        print("Loading image db from", image_db_dir)
        image_db = torch.load(image_db_dir)
    else:
        # create image db
        image_db = []
        for sample in ds:
            global_set = sample[0] # 画像と説明文のセット
            partial_sets = sample[1:] # 部分画像と説明文のセット（複数）
            
            g_img = global_set["image"]
            img = img_preprocess(Image.fromarray(g_img))
            img = img.unsqueeze(0).to(device)
            img_feat = model.encode_image(img)
            image_db.append(img_feat.to("cpu"))
        image_db = torch.cat(image_db, dim=0)
        torch.save(image_db, image_db_dir)
    print("image_db.shape", image_db.shape)
    return image_db

image_db_dir = args.dataset_type + "_img_db"
image_db = load_image_db(image_db_dir, ds, model, img_preprocess, device)

def load_text_db(text_db_dir, ds, model, device):
    if os.path.exists(text_db_dir):
        print("Loading text db from", text_db_dir)
        text_db = torch.load(text_db_dir)
    else:
        # create text db
        text_db = []
        for sample in ds:
            global_set = sample[0] # 画像と説明文のセット
            partial_sets = sample[1:] # 部分画像と説明文のセット（複数）

            g_caption = global_set["caption"]
            texts = clip.tokenize(g_caption, truncate=True).to(device)
            text_feat = model.encode_text(texts)
            text_db.append(text_feat.to("cpu"))
        text_db = torch.cat(text_db, dim=0)
        torch.save(text_db, text_db_dir)
    print("text_db.shape", text_db.shape)
    return text_db

text_db_dir = args.dataset_type + "_text_db"
text_db = load_text_db(text_db_dir, ds, model, device)

# global image -- global text
def feature_extraction(ds, model, img_preprocess, device):

    global_text_feats = []
    global_img_feats = []
    img_group_ids = []
    txt_group_ids = []

    i = 0
    for sample in tqdm(ds): # recall text->image
        with torch.no_grad():
            global_set = sample[0] # 画像と説明文のセット
            partial_sets = sample[1:] # 部分画像と説明文のセット（複数）

            g_img =global_set["image"] # 画像 (height, width, 3)
            g_caption =global_set["caption"] # 最初の説明文のみ
            # g_key =global_set["key"] # 画像のID？
            # g_captions =global_set["captions"] # 5種類全ての説明文
            # g_negative =global_set["negative"] # 負例の説明文？
            
            img_group_ids.append(i)
            txt_group_ids.append(i)
            # extract features
            img = img_preprocess(Image.fromarray(g_img)).unsqueeze(0).to(device)
            img_feat = model.encode_image(img).to("cpu")
            global_img_feats.append(img_feat)
            
            texts = clip.tokenize(g_caption, truncate=True).to(device)
            text_feat = model.encode_text(texts).to("cpu")  
            global_text_feats.append(text_feat)

        i += 1
    global_text_feats = torch.cat(global_text_feats, dim=0)
    global_img_feats = torch.cat(global_img_feats, dim=0)
    return global_text_feats, global_img_feats, img_group_ids, txt_group_ids

global_text_feats, global_img_feats, img_group_ids, txt_group_ids = feature_extraction(ds, model, img_preprocess, device)

global_text_feats = global_text_feats.float()
global_img_feats = global_img_feats.float()
image_db = image_db.float()
text_db = text_db.float()


def IPOT(query, db, n_iter=50, beta=0.5, save_dir=None, query_group_ids=None, db_group_ids=None):
    from copy import deepcopy
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt

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


# Recall_at_K
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
Text2Img_sim = torch.matmul(global_text_feats, image_db.T) 
Img2Text_sim = torch.matmul(global_img_feats, text_db.T) 
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

    Text2Img_OT_sim = IPOT(
        query=global_text_feats, 
        db=image_db,
        save_dir=save_dir_T2I, 
        query_group_ids=None, 
        db_group_ids=None)

    Img2Text_OT_sim = IPOT(
        query=global_img_feats, 
        db=text_db, 
        save_dir=save_dir_I2T, 
        query_group_ids=None, 
        db_group_ids=None)

    recall_at_k(Text2Img_OT_sim)
    recall_at_k(Img2Text_OT_sim)