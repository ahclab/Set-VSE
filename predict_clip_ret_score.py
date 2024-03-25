#!/usr/bin/env python

import os
import argparse
import random
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# from dataset_coco_small import CocoCaptions

def load_clip(model_name="ViT-B/32", device="cuda:0"):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, img_preprocess = clip.load(model_name, device=device)

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model, img_preprocess

# Set utility
tokenizer = _Tokenizer()

def id2vocab(tokenizer, tokens, include_special_tokens=False):
    """
    tokens: (77)
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.numpy()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    if not include_special_tokens:
        if tokens[0] == sot_token:
            tokens = tokens[1:]
        if eot_token in tokens:
            for idx, token in enumerate(tokens):
                if token == eot_token:
                    tokens = tokens[:idx]
                    break
    #raw_text = "".join([tokenizer.decoder[w] for w in tokens.numpy()]).replace("</w>", " ")
    raw_text = tokenizer.decode(tokens)

    return raw_text


# CLIP prediction
def clip_pred(img_feat, cap_feat):
    """
    img: (B,3,H,W), preprocessed array
    caps: list of strings
    """
    logits_per_image, logits_per_text = model(img_feat, cap_feat)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs, logits_per_image, logits_per_text

class Recall_at_K:
    """
    query: (B,512)
    db: (N,512)
    db_id: (N)
    """
    def __init__(self, Ks=[1,5,10]):
        self.n_count = {K:0 for K in Ks}
        self.n_total = 0
        self.max_K = max(Ks)

    def __call__(self, query, db, query_ids, db_ids):
        # accumulate total number of queries that satisfy the condition

        # cosine similarity
        sim = torch.matmul(db, query.T) # (N,B)
        # ranking
        ordered_index = torch.argsort(sim, dim=0, descending=True) # (N,B)

        # Extract the index of query_id that matches db_id 
        ordered_index = ordered_index.cpu()
        query_ids = query_ids
        ordered_db_ids = db_ids[ordered_index].cpu()
        condition = (ordered_db_ids == query_ids)
        assert torch.any(condition, dim=0).all() == True, "No query_id matches db_id"
        query_id_ranks = torch.argmax(condition*1.0, dim=0) # (B)

        # Calculate matching rate in image_id and query_id on each Top-K
        for k in self.n_count.keys():
            self.n_count[k] += torch.sum(query_id_ranks < k).item()
        self.n_total += query_ids.shape[0]

        # image_ids that are ranked higher than query_id
        higher_ranked_image_ids = {}
        for query_id, query_id_rank in zip(query_ids, query_id_ranks):
            higher_ranked_image_ids[query_id.item()] = ordered_db_ids[:query_id_rank].numpy().tolist()

        return higher_ranked_image_ids

        # # top K
        # ordered_index = ordered_index.cpu()
        # db_indice = ordered_index[:self.max_K, :] # (K,B)
        # topK_db_ids = db_ids[db_indice].T # (B,K)

        # # Calculate matching rate in image_id and query_id
        # for query_id, topK_db_id in zip(query_ids, topK_db_ids):
        #     for k in self.n_count.keys():
        #         if query_id in topK_db_id[:k]:
        #             self.n_count[k] += 1
        # self.n_total += query_ids.shape[0]

    def get_recall(self):
        recall = {K: self.n_count[K]/self.n_total for K in self.n_count.keys()}
        return recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_db_dir", type=str, default="./image_db")
    parser.add_argument("--subset_path", type=str, default="../mscoco_subsets/mscoco_val_sub0.json")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    print(vars(args))

    # fix random seed =====================
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def Worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    # ====================================

    # Make output dir
    if args.output_dir is None:
        path = Path(args.subset_path)
        output_dir = str(path.parent)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Load image db
    print("Loading image db dir:", args.image_db_dir)
    assert os.path.exists(args.image_db_dir)==True, "Image db dir does not exist"
    p = Path(args.image_db_dir)
    lis = list(p.glob("*.pt"))
    assert len(lis) > 0, "Image db dir is empty"
    img_dbs = []
    for path in lis:
        img_dbs.append(torch.load(str(path)))
    # stack
    img_db = {"img_feat": torch.cat([img_db["img_feat"] for img_db in img_dbs]),
               "img_id": torch.cat([img_db["img_id"] for img_db in img_dbs])}

    # Load CLIP model
    print("Loading CLIP model:", args.model_name)
    device = args.device
    model, img_preprocess = load_clip(args.model_name, device=device)
    print("Loaded")

    dataset = CocoCaptions(
        root = '/ahc/work4/seitaro-s/DATASETS/MSCOCO/images/val2014',
        ann_dict_path = args.subset_path,
        transform=img_preprocess
        )
    print("Dataset length:", len(dataset))
    eval_loader = DataLoader(dataset,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=args.n_cpu,
                               pin_memory=True,
                               drop_last=False,
                               worker_init_fn=Worker_init_fn,
                               generator=g,
                               )
    
    cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    i = 0
    rk = Recall_at_K(Ks=[1,5,10,100,1000,len(dataset)])
    higher_ranked_image_ids = {}
    token_lengths = []
    for samples in tqdm(eval_loader, desc="Eval iteration"):
        image, text, img_id, text_id = samples  
        with torch.no_grad():
            # imgs = image.to(device)

            # truncate tokens
            texts = clip.tokenize(text, truncate=True).to(device) # list -> torch tensor
            
            # labels = torch.arange(image.shape[0]).to(device)
            
            token_lengths += [len(tokenizer.encode(text_)) for text_ in text] # token length without <s> and </s>

            for text_ in text:
                if len(tokenizer.encode(text_)) < 26:
                    print("shorter text sample less than 26:", text_)
            # extract features
            # img_feat = model.encode_image(imgs)
            text_feat = model.encode_text(texts)

            # ranking images
            db_id = img_db["img_id"].to(device)
            db_feat = img_db["img_feat"].to(device)
            
            higher_ranked_image_ids_ = rk(text_feat, db_feat, img_id, db_id)
            for query_id, higher_ranked_image_id in higher_ranked_image_ids_.items():
                assert query_id not in higher_ranked_image_ids.keys(), "query_id already exists"
                higher_ranked_image_ids[query_id] = higher_ranked_image_id

        i += 1
    recall = rk.get_recall()

    with open(os.path.join(output_dir, "higher_ranked_image_ids.json"), "w") as f:
        json.dump(higher_ranked_image_ids, f)
    print(",".join(["{}".format(key) for key in recall.keys()]))
    print("-----") 
    print(",".join(["{:1.3f}".format(value) for value in recall.values()]))
    with open(os.path.join(output_dir, "recall.txt"), "w") as f:
        f.write(",".join(["{}".format(key) for key in recall.keys()])+"\n")
        f.write(",".join(["{:1.3f}".format(value) for value in recall.values()])+"\n")

    print("token lengths:")
    print(token_lengths)
    print("mean token length:", np.mean(token_lengths))
    print("max token length:", np.max(token_lengths))
    print("min token length:", np.min(token_lengths))
    with open(os.path.join(output_dir, "token_lengths.txt"), "w") as f:
        f.write("mean token length: {}\n".format(np.mean(token_lengths)))
        f.write("max token length: {}\n".format(np.max(token_lengths)))
        f.write("min token length: {}\n".format(np.min(token_lengths)))
        f.write("token lengths:\n")
        f.write(",".join([str(length) for length in token_lengths])+"\n")