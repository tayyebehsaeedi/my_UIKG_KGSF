#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
import random
from tqdm import tqdm
from math import exp
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
######################################## Edited by T.S based on system availability
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # for single-GPU system
########################################
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset,CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu

#################################### (Added by T.S)
import nltk
nltk.download('punkt_tab')  # For word tokenization
# nltk.download('punkt')  # For word tokenization
####################################

#################################### Define device as a global variable for set cuda is available else cpu (Added by T.S)
from config import DEVICE
####################################
#################################### Import another recommendatin metrics from CRSLab by T.S
from CRSLab_metrics import *
from CRSLab_metrics.rec import *
####################################
#################################### To Get Time
import time
####################################
#################################### Define random seed for generate same random variable in each run (Added by T.S)
from config import set_seed
####################################
# --- ADDED: JSON-output utilities and outputs layout (Option 2 structure) ---

def ensure_outputs(base='outputs'):
    """
    Create outputs folders:
      outputs/valid_rec, outputs/test_rec
      outputs/valid_gen, outputs/test_gen
    """
    os.makedirs(base, exist_ok=True)
    for split in ['valid', 'test']:
        os.makedirs(os.path.join(base, f"{split}_rec"), exist_ok=True)
        os.makedirs(os.path.join(base, f"{split}_gen"), exist_ok=True)
    return base

def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# def save_rec_outputs(base_out, split, pred_idx_list, rec_metrics):
def save_rec_outputs(base_out, split, pred_idx_list, conversation_ids, target_users, similar_users, similar_user_entity_ids, similar_user_scores, rec_metrics):
    
    """
    Save recommendation outputs:
      - pred_idx_list: list[list[int]] (movie IDs)
      - rec_metrics: dict
    Files:
      outputs/{split}_rec/pred_idx.json
      outputs/{split}_rec/metrics.json
    """
    """
    Save one record for each CRS sample.

    conversationId:
        Original conversationId from valid_data.jsonl/test_data.jsonl.

    target_user:
        Original ReDial user ID.

    similar_users:
        Original ReDial user IDs.

    similar_user_entity_ids:
        Corresponding UIKG entity IDs.

    similarity_scores:
        Cosine similarities between the conversation context
        and the retrieved users.

    pred_movies:
        Predicted movie IDs.
    """
    out_dir = os.path.join(base_out, f"{split}_rec")
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "pred_idx.json"), {"pred_idx": pred_idx_list})
    ############# save top-k similar user #############
    save_json(
        os.path.join(out_dir, "similar_users.json"),
        {
            "similar_users": similar_users,
            "similarity_scores": similar_user_scores
        }
    )
    ###################################################
    save_json(os.path.join(out_dir, "metrics.json"), rec_metrics)

    ########### save conversation id with target user, similar users, pred movies ###########
    conversations = []

    # Safety check
    n = len(pred_idx_list)

    assert len(conversation_ids) == n, (
        f"conversation_ids ({len(conversation_ids)}) != "
        f"pred_movies ({n})"
    )

    assert len(target_users) == n, (
        f"target_users ({len(target_users)}) != "
        f"pred_movies ({n})"
    )

    assert len(similar_users) == n, (
        f"similar_users ({len(similar_users)}) != "
        f"pred_movies ({n})"
    )

    assert len(similar_user_scores) == n, (
        f"similarity_scores ({len(similar_user_scores)}) != "
        f"pred_movies ({n})"
    )

    for idx in range(n):

        conversations.append({
            # Actual ReDial conversation ID
            "conversationId": str(conversation_ids[idx]),

            # Original ReDial target user ID
            "target_user": int(target_users[idx]),

            # Original ReDial similar user IDs
            "similar_users": [
                int(x)
                for x in similar_users[idx]
            ],

            # UIKG IDs corresponding to similar_users
            "similar_user_entity_ids": [
                int(x)
                for x in similar_user_entity_ids[idx]
            ],

            # Cosine similarity
            "similarity_scores": [
                float(x)
                for x in similar_user_scores[idx]
            ],

            # Predicted movie IDs
            "pred_movies": [
                int(x)
                for x in pred_idx_list[idx]
            ]
        })

    save_json(
        os.path.join(
            out_dir,
            "conversation_recommendations.json"
        ),
        conversations
    )
    #########################################################################################

    print(f"[saved rec outputs] {out_dir}")

def save_gen_outputs(base_out, split, contexts, generated, gold, gen_metrics):
    """
    Save generation outputs in Format B:
      payload = {
        "context": [...],    # list of token lists (strings)
        "generated": [...],
        "gold": [...],
        "metrics": { ... }
      }
    Saves to outputs/{split}_gen/generated.json and metrics.json
    """
    out_dir = os.path.join(base_out, f"{split}_gen")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "context": contexts,
        "generated": generated,
        "gold": gold,
        "metrics": gen_metrics
    }
    save_json(os.path.join(out_dir, "generated.json"), payload)
    save_json(os.path.join(out_dir, "metrics.json"), gen_metrics)
    # also write text copies for quick inspection (optional)
    with open(os.path.join(out_dir, "context.txt"), "w", encoding="utf-8") as f:
        f.writelines([" ".join(s) + "\n" for s in contexts])
    with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
        f.writelines([" ".join(s) + "\n" for s in generated])
    with open(os.path.join(out_dir, "gold.txt"), "w", encoding="utf-8") as f:
        f.writelines([" ".join(s) + "\n" for s in gold])
    print(f"[saved gen outputs] {out_dir}")
# --- end added utilities ---



def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-batch_size","--batch_size",type=int,default=32)
    train.add_argument("-max_count","--max_count",type=int,default=5)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-epoch","--epoch",type=int,default=10)
    train.add_argument("-gpu","--gpu",type=str,default='0,1')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    train.add_argument("-embedding_size","--embedding_size",type=int,default=300)

    train.add_argument("-n_heads","--n_heads",type=int,default=2)
    train.add_argument("-n_layers","--n_layers",type=int,default=2)
    train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

    train.add_argument("-dropout","--dropout",type=float,default=0.1)
    train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
    train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

    train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

    train.add_argument("-n_entity","--n_entity",type=int,default=31299)
    train.add_argument("-n_relation","--n_relation",type=int,default=35)
    train.add_argument("-n_concept","--n_concept",type=int,default=29308)
    train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
    train.add_argument("-dim","--dim",type=int,default=128)
    train.add_argument("-n_hop","--n_hop",type=int,default=2)
    train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
    train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
    train.add_argument("-n_memory","--n_memory",type=float,default=32)
    train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
    train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)
    train.add_argument("-rel_threshold", "--rel_threshold", type=int, default=1000)

    ################################################################################
    train.add_argument("-seed", "--seed", type=int, default=42)
    train.add_argument("-fu_epoch", "--fu_epoch", type=int, default=7)
    train.add_argument("-sl_id", "--sl_id", type=int, default=34)
    train.add_argument("-path_KG", "--path_KG", type=str, default='data/UIKG/v2/')
    train.add_argument("-KG_name", "--KG_name", type=str, default='uikg.pkl')
    train.add_argument("-weight_decay", "--weight_decay", type=float, default=1e-3, help="Weight decay for AdamW")
    ################################################################################
    train.add_argument("--top_k_users", type=int, default=20)
    train.add_argument("--neighbor_weight", type=float, default=0.2)
    train.add_argument("--use_seen_dislike_neighbor_ablation", action="store_true",
                       help="Expand neighbor candidates with seen movies and apply weighted neighbor-dislike penalties.")
    train.add_argument("--neighbor_like_weight", type=float, default=1.0)
    train.add_argument("--neighbor_seen_weight", type=float, default=0.2)
    train.add_argument("--neighbor_dislike_weight", type=float, default=1.0)
    train.add_argument("--neighbor_support_threshold", type=float, default=0.0)
    train.add_argument("--enable_attribute_expansion", action="store_true")
    train.add_argument("--person_weight", type=float, default=0.5,
                       help="Weight for movies linked to liked directors/writers/starring people.")
    train.add_argument("--genre_weight", type=float, default=0.4)
    train.add_argument("--disliked_genre_weight", type=float, default=0.2)

    return train

class TrainLoop_fusion_rec():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.path_KG = self.opt['path_KG']
        self.train_dataset=dataset('data/train_data.jsonl',opt)

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = pkl.load(open(self.path_KG + "movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        # self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        ################################################### Added another recommender metrics from CRSLab by T.S
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"MRR@1":0,"MRR@10":0,"MRR@50":0,"Hit@1":0,"Hit@10":0,"Hit@50":0,"NDCG@1":0,"NDCG@10":0,"NDCG@50":0,"loss":0,"count":0}
        # self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"MRR@1":0,"MRR@10":0,"MRR@50":0,"loss":0,"count":0}
        ###################################################
        # self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)

        # Add these lines right after build_model() call
        self.generator = torch.Generator()
        self.generator.manual_seed(self.opt['seed'])

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

        ###########################################################################
        self.seed = self.opt['seed']
        self.fu_epoch = self.opt['fu_epoch']
        ###########################################################################
        # ADDED: outputs base dir and accumulators for recommendation
        self.base_out = ensure_outputs(self.path_KG + 'outputs')
        # collect mapped movie ids across validation/test for saving
        self.pred_idx_all = []  # list of lists (movie ids)

        ####################### save top_k users and scores #######################
        self.conversation_id_all = []     # Original ReDial conversation IDs
        self.similar_user_all = []        # Original ReDial similar user IDs
        self.similar_user_entity_all = [] # UIKG entity IDs (kept for debugging)
        self.similar_user_scores_all = [] # Cosine similarities
        self.target_user_all = []         # Original ReDial target user IDs
        ###########################################################################                


    def build_model(self,is_finetune):
        ###########################################################################
        set_seed()
        ###########################################################################

        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    #########################################################################
    @staticmethod
    def seed_worker(worker_id):
        """Ensure each worker has a different but deterministic seed"""
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    #########################################################################

    def train(self):
        #self.model.load_model()
        ###########################################################################
        set_seed(self.seed)
        ###########################################################################

        losses=[]
        best_val_rec=0
        rec_stop=False
        # for i in range(3):
        for i in tqdm(range(self.fu_epoch), desc= "Pre-trainin Epochs"):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
            # train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'], self.opt['seed'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            worker_init_fn=self.seed_worker,
                                                            generator=self.generator)
            # train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
            #                                                 batch_size=self.batch_size,
            #                                                 shuffle=False)
            num=0
            # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec in tqdm(train_dataset_loader):
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,uikg_mask,concept_vec, uikg_vec,rec, conversation_id,target_user_node,history_seen,history_disliked in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                # scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(),
                #                                                                                                             concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, test=False)
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(),
                                                                                                                            concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec,
                                                                                                                            target_user_nodes=target_user_node.cuda(), history_seen_nodes=history_seen.cuda(), history_disliked_nodes=history_disliked.cuda(), test=False)

                joint_loss=info_uikg_loss#+info_con_loss

                losses.append([info_uikg_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('info uikg loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    #print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

        print("masked loss pre-trained")
        losses=[]

        for i in tqdm(range(self.epoch), desc= "Main Recommendation Training"):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
            # train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'], self.opt['seed'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            worker_init_fn=self.seed_worker,
                                                            generator=self.generator)
            num=0
            # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,uikg_mask,concept_vec, uikg_vec,rec in tqdm(train_dataset_loader, desc=f"Batches training on {i+1} Epoch"):
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,uikg_mask,concept_vec, uikg_vec,rec,conversation_id,target_user_node,history_seen,history_disliked in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                # scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie,concept_vec, uikg_vec, entity_vector.cuda(), rec, test=False)
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, _=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie,concept_vec, uikg_vec, entity_vector.cuda(), rec,
                                                                                                       target_user_nodes=target_user_node.cuda(), history_seen_nodes=history_seen.cuda(), history_disliked_nodes=history_disliked.cuda(), test=False)

                joint_loss=rec_loss+0.025*info_uikg_loss#+0.0*info_con_loss#+mask_loss*0.05

                losses.append([rec_loss,info_uikg_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('rec loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('info uikg loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

            output_metrics_rec = self.val()

            if best_val_rec > output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]:
                rec_stop=True
            else:
                best_val_rec = output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]
                self.model.save_model()
                print("recommendation model saved once------------------------------------------------")

            if rec_stop==True:
                break

        _=self.val(is_test=True)
        # # Save the exact parameters used for this evaluation Added by T.S for saving last parameters not just who have better results on validation
        # rec_path = os.path.join("saved_model", "last_rec_for_eval.pth")
        # torch.save(self.model.state_dict(), rec_path)
        # print(f"✅ Last recommender model (used for final evaluation) saved to '{rec_path}'")

    def metrics_cal_rec(self,rec_loss,scores,labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        ####################################################################
        # Map pred_idx (indices into movie_ids) to actual movie IDs
        movie_ids_tensor = torch.LongTensor(self.movie_ids)
        pred_idx_movieids = []
        for b in range(pred_idx.size(0)):
            row = pred_idx[b].tolist()
            mapped = [int(movie_ids_tensor[i]) for i in row]
            pred_idx_movieids.append(mapped)
        # Append to the global collector (flat across batches)
        if not hasattr(self, "pred_idx_all"):
            self.pred_idx_all = []
        self.pred_idx_all.extend(pred_idx_movieids)
        ####################################################################
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            ################################################# Added another recommender metrics
            # breakpoint()
            self.metrics_rec["MRR@1"] += MRRMetric.compute(pred_idx[b].tolist(), target_idx, 1).value()
            self.metrics_rec["MRR@10"] += MRRMetric.compute(pred_idx[b].tolist(), target_idx, 10).value()
            self.metrics_rec["MRR@50"] += MRRMetric.compute(pred_idx[b].tolist(), target_idx, 50).value()
            # breakpoint()
            self.metrics_rec["Hit@1"] += HitMetric.compute(pred_idx[b].tolist(), target_idx, 1).value()
            self.metrics_rec["Hit@10"] += HitMetric.compute(pred_idx[b].tolist(), target_idx, 10).value()
            self.metrics_rec["Hit@50"] += HitMetric.compute(pred_idx[b].tolist(), target_idx, 50).value()

            self.metrics_rec["NDCG@1"] += NDCGMetric.compute(pred_idx[b].tolist(), target_idx, 1).value()
            self.metrics_rec["NDCG@10"] += NDCGMetric.compute(pred_idx[b].tolist(), target_idx, 10).value()
            self.metrics_rec["NDCG@50"] += NDCGMetric.compute(pred_idx[b].tolist(), target_idx, 50).value()
            ##################################################
            self.metrics_rec["count"] += 1

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        # self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        ################################################ Added another recommender metrics from CRSLab by T.S
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"MRR@1":0,"MRR@10":0,"MRR@50":0,"Hit@1":0,"Hit@10":0,"Hit@50":0,"NDCG@1":0,"NDCG@10":0,"NDCG@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        ################################################
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        # val_set=CRSdataset(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'], self.opt['seed'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            worker_init_fn=self.seed_worker,
                                                            generator=self.generator)
        # val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
        #                                                    batch_size=self.batch_size,
        #                                                    shuffle=False)
        recs=[]
        # for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, uikg_mask, concept_vec, uikg_vec, rec in tqdm(val_dataset_loader):
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, uikg_mask, concept_vec, uikg_vec, rec, conversation_id, target_user_node, history_seen, history_disliked in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                # scores, preds, rec_scores, rec_loss, _, mask_loss, info_uikg_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_uikg_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, target_user_nodes=target_user_node.cuda(), history_seen_nodes=history_seen.cuda(), history_disliked_nodes=history_disliked.cuda(), test=True, maxlen=20, bsz=batch_size)

                ###################################################################
                self.conversation_id_all.extend(
                    conversation_id
                )
                # ---------------------------------------------------------
                # Collect target user IDs
                # ---------------------------------------------------------
                target_user_redial = (
                    target_user_node - self.model.user_entity_offset
                )
                self.target_user_all.extend(
                    target_user_redial.cpu().tolist()
                )
                # ---------------------------------------------------------
                # Collect top-k similar users
                # ---------------------------------------------------------
                self.similar_user_all.extend(self.model.last_similar_user_ids)
                self.similar_user_entity_all.extend(self.model.last_similar_user_entity_ids)
                self.similar_user_scores_all.extend(self.model.last_similar_user_scores)
                ###################################################################

            recs.extend(rec.cpu())
            #print(losses)
            #exit()
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        print(output_dict_rec)

        # --- ADDED: save recommendation outputs (only top-100 pred idxs and metrics) ---
        # Prepare a JSON-safe copy of metrics_rec
        rec_metrics_to_save = {
            key: (value.item() if isinstance(value, torch.Tensor) else value)
            for key, value in output_dict_rec.items()
        }
        split_name = "test" if is_test else "valid"
        # pred_idx_all was collected during batch processing
        pred_idx_list = getattr(self, "pred_idx_all", [])
        ##################### save top-k users #####################
        conversation_id_list = getattr(self, "conversation_id_all", [])
        target_user_list = getattr(self, "target_user_all", [])

        similar_user_list = getattr(self, "similar_user_all", [])

        similar_user_entity_list = getattr(self, "similar_user_entity_all", [])

        similar_user_scores = getattr(self, "similar_user_scores_all", [])
        ############################################################
        # Save to outputs/<split>_rec
        # save_rec_outputs(self.base_out, split_name, pred_idx_list, rec_metrics_to_save)
        save_rec_outputs(
            self.base_out,
            split_name,
            pred_idx_list,
            conversation_id_list,
            target_user_list,
            similar_user_list,
            similar_user_entity_list,
            similar_user_scores,
            rec_metrics_to_save
        )
        # Clear the collector for next eval
        self.pred_idx_all = []
        self.conversation_id_all = []
        self.target_user_all = []
        self.similar_user_all = []
        self.similar_user_entity_all = []
        self.similar_user_scores_all = []
        # --- end added ---


        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        
        # Add AdamW explicitly
        from torch.optim import AdamW
        optims['adamw'] = AdamW

        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}

        optimizer_name = opt['optimizer'].lower()
        if optimizer_name in ['adam', 'adamw']:
            kwargs['weight_decay'] = opt.get('weight_decay', 0.01)  # AdamW uses weight_decay, Adam usually 0
            if optimizer_name == 'adamw':
                kwargs['weight_decay'] = opt.get('weight_decay', 0.01)  # typical: 0.01
            else:
                kwargs['weight_decay'] = 0.0  # standard Adam doesn't use L2

        if optimizer_name == 'adam':
            kwargs['amsgrad'] = True  # optional, paper uses default Adam → usually False, but code had True

        # kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

class TrainLoop_fusion_gen():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.path_KG = self.opt['path_KG']
        self.train_dataset=dataset('data/train_data.jsonl',opt)

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = pkl.load(open(self.path_KG + "movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        # self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        ################################################### Added another recommender metrics from CRSLab by T.S
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"MRR@1":0,"MRR@10":0,"MRR@50":0,"loss":0,"count":0}
        ###################################################
        # self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune=True)

        # Add these lines right after build_model() call
        self.generator = torch.Generator()
        self.generator.manual_seed(self.opt['seed'])

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

        ###########################################################################
        self.seed = self.opt['seed']
        ###########################################################################
        # ADDED: outputs base dir and accumulators for generation
        self.base_out = ensure_outputs(self.path_KG + 'outputs')
        # generation accumulators will be local to val() runs (no global store needed)


    def build_model(self,is_finetune):
        ###########################################################################
        set_seed()
        ###########################################################################

        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    ###############################################################################
    @staticmethod
    def seed_worker(worker_id):
        """Ensure each worker has a different but deterministic seed"""
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    ###############################################################################

    def train(self):
        ###########################################################################
        set_seed(self.seed)
        ###########################################################################

        self.model.load_model()
        losses=[]
        best_val_gen=1000
        gen_stop=False
        for i in tqdm(range(self.epoch*3), desc= "Main Generation Training"):
        # for i in tqdm(range(self.epoch*2), desc= "Main Generation Training"):
            train_set=CRSdataset(self.train_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
            # train_set=CRSdataset(self.train_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'], self.opt['seed'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            worker_init_fn=self.seed_worker,
                                                            generator=self.generator)
            # train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
            #                                                 batch_size=self.batch_size,
            #                                                 shuffle=False)
            num=0
            # for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,uikg_mask,concept_vec, uikg_vec,rec in tqdm(train_dataset_loader, desc=f"Batches training on {i+1} Epoch"):
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,uikg_mask,concept_vec, uikg_vec,rec,conversation_id,target_user_node,history_seen,history_disliked in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                # scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, info_con_loss=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, test=False)
                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, info_con_loss=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, target_user_nodes=target_user_node.cuda(), history_seen_nodes=history_seen.cuda(), history_disliked_nodes=history_disliked.cuda(), test=False)

                joint_loss=gen_loss

                losses.append([gen_loss])
                self.backward(joint_loss)
                self.update_params()
                # if num%50==0:
                if num%100==0:
                    print('gen loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    losses=[]
                num+=1

            output_metrics_gen = self.val(True)
            if best_val_gen < output_metrics_gen["dist4"]:
                pass
            else:
                best_val_gen = output_metrics_gen["dist4"]
                self.model.save_model()
                print("generator model saved once------------------------------------------------")

        _=self.val(is_test=True)
        # # Save the exact parameters used for this evaluation Added by T.S for saving last parameters not just who have better results on validation
        # gen_path = os.path.join("saved_model", "last_gen_for_eval.pth")
        # torch.save(self.model.state_dict(), gen_path)
        # print(f"✅ Last generator model (used for final evaluation) saved to '{gen_path}'")

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        # self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        ################################################### Added another recommender metrics from CRSLab by T.S
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"MRR@1":0,"MRR@10":0,"MRR@50":0,"Hit@1":0,"Hit@10":0,"Hit@50":0,"NDCG@1":0,"NDCG@10":0,"NDCG@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        ###################################################
        self.model.eval()
        ################################################### For computing th perplexity (added by T.S)
        # # reset PPL accumulators
        # self.total_loss_for_ppl = 0.0  # PPL: sum of NLL over all tokens
        # self.total_tokens_for_ppl = 0  # PPL: total number of tokens
        ###################################################
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
        # val_set=CRSdataset(val_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'], self.opt['seed'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            worker_init_fn=self.seed_worker,
                                                            generator=self.generator)
        # val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
        #                                                    batch_size=self.batch_size,
        #                                                    shuffle=False)
        inference_sum=[]
        golden_sum=[]
        context_sum=[]
        losses=[]
        recs=[]
        # for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, uikg_mask, concept_vec, uikg_vec, rec in tqdm(val_dataset_loader):
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, uikg_mask, concept_vec, uikg_vec, rec, conversation_id, target_user_node, history_seen, history_disliked in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                # _, _, _, _, gen_loss, mask_loss, info_uikg_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, test=False)
                # scores, preds, rec_scores, rec_loss, _, mask_loss, info_uikg_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)
                _, _, _, _, gen_loss, mask_loss, info_uikg_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, target_user_nodes=target_user_node.cuda(), history_seen_nodes=history_seen.cuda(), history_disliked_nodes=history_disliked.cuda(), test=False)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_uikg_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, uikg_mask, seed_sets, movie, concept_vec, uikg_vec, entity_vector.cuda(), rec, target_user_nodes=target_user_node.cuda(), history_seen_nodes=history_seen.cuda(), history_disliked_nodes=history_disliked.cuda(), test=True, maxlen=20, bsz=batch_size)

            golden_sum.extend(self.vector2sentence(response.cpu()))
            inference_sum.extend(self.vector2sentence(preds.cpu()))
            context_sum.extend(self.vector2sentence(context.cpu()))
            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            #print(losses)
            #exit()
            ################################################# For computing th perplexity (added by T.S)
            # # PPL: accumulate token-level loss
            # batch_tokens = response.ne(0).sum().item()
            # self.total_loss_for_ppl += gen_loss.sum().item()
            # self.total_tokens_for_ppl += batch_tokens
            #################################################

        self.metrics_cal_gen(losses,inference_sum,golden_sum,recs)

        ################################################# For computing th perplexity (added by T.S)
        # # PPL: compute final perplexity
        # if self.total_tokens_for_ppl > 0:
        #     import math
        #     self.metrics_gen["ppl"] = math.exp(self.total_loss_for_ppl / self.total_tokens_for_ppl)
        # else:
        #     self.metrics_gen["ppl"] = float("inf")
        #################################################

        output_dict_gen={}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['count']
            else:
                output_dict_gen[key]=self.metrics_gen[key]
        print(output_dict_gen)

        # f=open('context_test.txt','w',encoding='utf-8')
        f=open(self.path_KG + 'outputs/'+'context_test.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in context_sum])
        f.close()

        # f=open('output_test.txt','w',encoding='utf-8')
        f=open(self.path_KG + 'outputs/'+'output_test.txt','w',encoding='utf-8')
        f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
        f.close()

        # --- ADDED: Save generation outputs (context, generated, gold) in Format B ---
        split_name = "test" if is_test else "valid"

        contexts_out = context_sum      # list[list[str]]
        generated_out = inference_sum   # list[list[str]]
        gold_out = golden_sum           # list[list[str]]

        # Merge generation metrics into a dict (already done in output_dict_gen)
        gen_metrics = output_dict_gen

        # Save to outputs/{split}_gen
        save_gen_outputs(self.base_out, split_name, contexts_out, generated_out, gold_out, gen_metrics)
        # --- end added ---

        return output_dict_gen

    def metrics_cal_gen(self,rec_loss,preds,responses,recs):
        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4

        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count=0
            quagram_count=0
            unigram_set = set()
            bigram_set = set()
            trigram_set=set()
            quagram_set=set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count+=1
                    trigram_set.add(trg)
                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_count+=1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(outs)#unigram_count
            dis2 = len(bigram_set) / len(outs)#bigram_count
            dis3 = len(trigram_set)/len(outs)#trigram_count
            dis4 = len(quagram_set)/len(outs)#quagram_count
            return dis1, dis2, dis3, dis4

        predict_s=preds
        golden_s=responses
        print(rec_loss[0])
        self.metrics_gen["ppl"]+=sum([exp(ppl) for ppl in rec_loss])/len(rec_loss)
        generated=[]

        for out, tar, rec in zip(predict_s, golden_s, recs):
            bleu1, bleu2, bleu3, bleu4=bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen['bleu1']+=bleu1
            self.metrics_gen['bleu2']+=bleu2
            self.metrics_gen['bleu3']+=bleu3
            self.metrics_gen['bleu4']+=bleu4
            self.metrics_gen['count']+=1

        dis1, dis2, dis3, dis4=distinct_metrics(generated)
        self.metrics_gen['dist1']=dis1
        self.metrics_gen['dist2']=dis2
        self.metrics_gen['dist3']=dis3
        self.metrics_gen['dist4']=dis4

    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.index2word[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        
        # Add AdamW explicitly
        from torch.optim import AdamW
        optims['adamw'] = AdamW

        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}

        optimizer_name = opt['optimizer'].lower()
        if optimizer_name in ['adam', 'adamw']:
            kwargs['weight_decay'] = opt.get('weight_decay', 0.01)  # AdamW uses weight_decay, Adam usually 0
            if optimizer_name == 'adamw':
                kwargs['weight_decay'] = opt.get('weight_decay', 0.01)  # typical: 0.01
            else:
                kwargs['weight_decay'] = 0.0  # standard Adam doesn't use L2

        if optimizer_name == 'adam':
            kwargs['amsgrad'] = True  # optional, paper uses default Adam → usually False, but code had True

        # kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

if __name__ == '__main__':
    args=setup_args().parse_args()
    print(vars(args))

    ###########################################################################
    set_seed(vars(args)['seed'])
    ###########################################################################

    if args.is_finetune==False:
        loop=TrainLoop_fusion_rec(vars(args),is_finetune=False)
        #loop.model.load_model()
        start_train = time.time()
        loop.train()
        end_train = time.time()
        print(f"Training time: {(end_train - start_train) / 60:.2f} minutes")
    else:
        loop=TrainLoop_fusion_gen(vars(args),is_finetune=True)
        #loop.train()
        loop.model.load_model()
        #met = loop.val(True)
        start_train_gen = time.time()
        loop.train()
        end_train_gen = time.time()
        print(f"Training time for generating module: {(end_train_gen - start_train_gen) / 60:.2f} minutes")

    start_test = time.time()
    met=loop.val(True)
    end_test = time.time()
    print(f"Test time: {(end_test - start_test) / 60:.2f} minutes")
    #print(met)

    # --- ADDED: save a basic training log to outputs/training_log.json ---
    if args.is_finetune==False:
        try:
            os.makedirs(vars(args)['path_KG'] + 'outputs/rec', exist_ok=True)
            training_log = {
                "args": vars(args),
                "seed": vars(args).get("seed"),
                "finished_at": time.time(),
                "note": "See outputs/ for saved rec outputs."
            }
            save_json(os.path.join(vars(args)['path_KG'] + "outputs/rec", "training_log.json"), training_log)
            print("Saved training log to" + vars(args)['path_KG'] + "outputs/rec/training_log.json")
        except Exception as e:
            print("Failed saving training_log.json:", e)
    else:
        try:
            os.makedirs(vars(args)['path_KG'] + 'outputs/gen', exist_ok=True)
            training_log = {
                "args": vars(args),
                "seed": vars(args).get("seed"),
                "finished_at": time.time(),
                "note": "See outputs/ for saved gen outputs."
            }
            save_json(os.path.join(vars(args)['path_KG'] + "outputs/gen", "training_log.json"), training_log)
            print("Saved training log to" + vars(args)['path_KG'] + "outputs/gen/training_log.json")
        except Exception as e:
            print("Failed saving training_log.json:", e)

