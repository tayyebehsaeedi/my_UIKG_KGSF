from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json

#################################### Define device as a global variable for set cuda is available else cpu (Added by T.S)
from config import DEVICE
####################################

#################################### Define random seed for generate same random variable in each run (Added by T.S)
from config import set_seed
# import random
# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
####################################

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    ###################################################################################
    set_seed()
    ###################################################################################

    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop, threshold = 1000, sl_id = 185):
    ###################################################################################
    set_seed()
    ###################################################################################

    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            # edge_list.append((entity, entity, 185))
            edge_list.append((entity, entity, sl_id))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != sl_id :# and tail_and_relation[0] in EDGE_TYPES:
                # if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > threshold and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > threshold], len(relation_idx)

def _uikg_edge_list(kg):
    """Return all custom-KG edges without KGSF's DBpedia frequency pruning."""
    ###################################################################################
    set_seed()
    ###################################################################################
    edges = {(head, tail, relation)
             for head, neighbors in kg.items()
             for relation, tail in neighbors}
    if not edges:
        raise ValueError("uikg.pkl has no edges")
    relation_ids = {relation: index for index, relation in
                    enumerate(sorted({relation for _, _, relation in edges}))}
    return [(head, tail, relation_ids[relation]) for head, tail, relation in edges], len(relation_ids)

def concept_edge_list4GCN():
    ###################################################################################
    set_seed()
    ###################################################################################

    node2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
    f=open('conceptnet_edges2nd.txt',encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    # return torch.LongTensor(edge_set).cuda() # Original line based on torch 1.3.0 (avoid this)
    ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
    return torch.tensor(edge_set, dtype=torch.long, device=DEVICE)


class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        # self.pad_idx = dictionary[dictionary.null_token]
        # self.start_idx = dictionary[dictionary.start_token]
        # self.end_idx = dictionary[dictionary.end_token]
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.path_KG = opt['path_KG']
        self.KG_name = opt['KG_name']
        ####### store the user Id same as in the ReDial #######
        # UIKG user entity IDs are created as:
        #     ReDial user ID + 1112
        # Therefore:
        #     ReDial user ID = UIKG user entity ID - 1112
        # Mapping between UIKG user entity IDs and original ReDial user IDs
        self.user_entity_offset = 1112
        ########################################################
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg = pkl.load(
            open(self.path_KG + self.KG_name, "rb")
        )
        ############## Get the hyperparameters of the find similar users and recommend movies
        ############## based on liked people and genres ######################################
        self.top_k_users = opt.get('top_k_users', 20)
        self.neighbor_weight = opt.get('neighbor_weight', 0.2)
        # Ablation disabled by default: baseline retrieves only neighbor likes.
        # add seen movies from similar user and remove disliked movies from similar user
        self.use_seen_dislike_neighbor_ablation = opt.get('use_seen_dislike_neighbor_ablation', False)
        self.neighbor_like_weight = opt.get('neighbor_like_weight', 1.0)
        self.neighbor_seen_weight = opt.get('neighbor_seen_weight', 0.2)
        self.neighbor_dislike_weight = opt.get('neighbor_dislike_weight', 1.0)
        self.neighbor_support_threshold = opt.get('neighbor_support_threshold', 0.0)
        # Ablation disabled by default: baseline retrieves only neighbor likes.
        # add movie from person and genres liked by similar user
        self.enable_attribute_expansion = opt.get('enable_attribute_expansion', False)
        self.person_weight = opt.get('person_weight', 0.5)
        self.genre_weight = opt.get('genre_weight', 0.4)
        self.disliked_genre_weight = opt.get('disliked_genre_weight', 0.2)
        ######################################################################################
        ######################### Load Files based users and movies ##########################
        metadata = pkl.load(open(self.path_KG + "processed/uikg_metadata.pkl", "rb"))
        if opt['n_entity'] != metadata['n_nodes'] + 6:
            raise ValueError(f"--n_entity must equal uikg_metadata.pkl['n_nodes'] + 6. Now --n_entity is {opt['n_entity']} while uikg_metadata.pkl['n_nodes'] is {metadata['n_nodes']+6}")
        self.user_liked_movies = pkl.load(open(self.path_KG + "processed/user_liked_movies.pkl", "rb"))
        # These are used only for similar-user aggregation in the optional
        # ablation; target-user filtering remains dialogue-history-only.
        self.neighbor_disliked_movies = pkl.load(open(self.path_KG + "processed/user_disliked_movies.pkl", "rb"))
        self.neighbor_seen_movies = pkl.load(open(self.path_KG + "processed/user_seen_movies.pkl", "rb"))
        self.user_liked_people = pkl.load(open(self.path_KG + "processed/user_liked_people.pkl", "rb"))
        self.user_liked_genres = pkl.load(open(self.path_KG + "processed/user_liked_genres.pkl", "rb"))
        self.user_disliked_genres = pkl.load(open(self.path_KG + "processed/user_disliked_genres.pkl", "rb"))
        self.movie_to_people = pkl.load(open(self.path_KG + "processed/movie_to_people.pkl", "rb"))
        self.movie_to_genres = pkl.load(open(self.path_KG + "processed/movie_to_genres.pkl", "rb"))
        self.person_to_movies = pkl.load(open(self.path_KG + "processed/person_to_movies.pkl", "rb"))
        self.genre_to_movies = pkl.load(open(self.path_KG + "processed/genre_to_movies.pkl", "rb"))
        self.register_buffer('uikg_user_nodes', torch.LongTensor(metadata['user_entity_ids']))
        self.register_buffer('uikg_movie_nodes', torch.LongTensor(metadata['movie_entity_ids']))
        #######################################################################

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.uikg_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.uikg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        # self.criterion = nn.CrossEntropyLoss(reduce=False) # Old (PyTorch 1.3.0 syntax - deprecated)
        ####################################################################
        self.criterion = nn.CrossEntropyLoss(reduction='none') # New (PyTorch 2.3.0+ syntax) added by T.S
        ####################################################################

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.self_attn_uikg = SelfAttentionLayer(opt['dim'], opt['dim'])

        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_uikg_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_uikg = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        # self.info_con_loss = nn.MSELoss(size_average=False,reduce=False) # Old (PyTorch 1.3.0 syntax - deprecated)
        ####################################################################
        self.info_con_loss = nn.MSELoss(reduction='none') # New (PyTorch 2.3.0+ syntax) added by T.S
        ####################################################################
        # self.info_db_loss = nn.MSELoss(size_average=False,reduce=False) # Old (PyTorch 1.3.0 syntax - deprecated)
        ####################################################################
        self.info_uikg_loss = nn.MSELoss(reduction='none') # New (PyTorch 2.3.0+ syntax) added by T.S
        ####################################################################

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)

        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        # edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        # edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2, threshold= opt['rel_threshold'], sl_id= opt['sl_id'])
        edge_list, self.n_relation = _uikg_edge_list(self.kg)
        edge_list = list(set(edge_list))
        print(len(edge_list), self.n_relation)
        # self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda() # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        self.uikg_edge_sets = torch.tensor(edge_list, dtype=torch.long, device=DEVICE)
        ###########################################################
        self.uikg_edge_idx = self.uikg_edge_sets[:, :2].t()
        self.uikg_edge_type = self.uikg_edge_sets[:, 2]

        self.uikg_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases'])
        #self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.n_con_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)

        #self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        w2i=json.load(open('word2index_redial.json',encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}

        # self.mask4key=torch.Tensor(np.load('mask4key.npy')).cuda() # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        self.mask4key = torch.from_numpy(np.load('mask4key.npy')).float().to(DEVICE)
        ##########################################################
        # self.mask4movie=torch.Tensor(np.load('mask4movie.npy')).cuda() # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        self.mask4movie = torch.from_numpy(np.load('mask4movie.npy')).float().to(DEVICE)
        ##########################################################
        self.mask4=self.mask4key+self.mask4movie
        if is_finetune:
            params = [self.uikg_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_uikg.parameters(), self.user_norm.parameters(),
                      self.gate_norm.parameters(), self.output_en.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False

        ###################################################################################
        self.seed = opt['seed']
        ###################################################################################

    def _starts(self, bsz):
        """Return bsz start tokens."""

        return self.START.detach().expand(bsz, 1)

    ################## Retrive similar user by cosine similarity ##################
    # def retrieve_similar_users(self, context_emb, user_embeddings, top_k=20):
    #     context_emb = F.normalize(context_emb, dim=-1)
    #     user_embeddings = F.normalize(user_embeddings, dim=-1)

    #     similarities = torch.matmul(context_emb, user_embeddings.t())
    #     top_scores, top_indices = torch.topk(similarities, k=top_k, dim=-1)

    #     return top_indices, top_scores
    ###############################################################################

    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_uikg, attention_kg, attention_uikg, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_uikg, incr_state)
            #batch*1*hidden
            scores = scores[:, -1:, :]
            #scores = self.output(scores)
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            
            uikg_attn_norm = self.uikg_attn_norm(attention_uikg)

            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), uikg_attn_norm.unsqueeze(1), scores], -1))

            # logits = self.output(latent)
            con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
            voc_logits = F.linear(scores, self.embeddings.weight)
            # print(logits.size())
            # print(mem_logits.size())
            #gate = F.sigmoid(self.gen_gate_norm(scores))

            sum_logits = voc_logits + con_logits #* (1 - gate)
            _, preds = sum_logits.max(dim=-1)
            #scores = F.linear(scores, self.embeddings.weight)

            #print(attention_map)
            #print(db_attention_map)
            #print(preds.size())
            #print(con_logits.size())
            #exit()
            #print(con_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            #print(voc_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            
            #print(torch.topk(voc_logits.squeeze(0).squeeze(0),k=50)[1])

            #sum_logits = scores
            # print(sum_logits.size())

            #_, preds = sum_logits.max(dim=-1)
            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_uikg, attention_kg, attention_uikg, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_uikg) #batch*r_l*hidden

        kg_attention_latent=self.kg_attn_norm(attention_kg)

        #map=torch.bmm(latent,torch.transpose(kg_embs_norm,2,1))
        #map_mask=((1-encoder_states_kg[1].float())*(-1e30)).unsqueeze(1)
        #attention_map=F.softmax(map*map_mask,dim=-1)
        #attention_latent=torch.bmm(attention_map,encoder_states_kg[0])

        uikg_attention_latent=self.uikg_attn_norm(attention_uikg)

        #db_map=torch.bmm(latent,torch.transpose(db_embs_norm,2,1))
        #db_map_mask=((1-encoder_states_db[1].float())*(-1e30)).unsqueeze(1)
        #db_attention_map=F.softmax(db_map*db_map_mask,dim=-1)
        #db_attention_latent=torch.bmm(db_attention_map,encoder_states_db[0])

        copy_latent=self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), uikg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), latent],-1))

        #logits = self.output(latent)
        con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)
        # print(logits.size())
        # print(mem_logits.size())
        #gate=F.sigmoid(self.gen_gate_norm(latent))

        sum_logits = logits+con_logits#*(1-gate)
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def infomax_loss(self, con_nodes_features, uikg_nodes_features, con_user_emb, uikg_user_emb, con_label, uikg_label, mask):
        #batch*dim
        #node_count*dim
        con_emb=self.info_con_norm(con_user_emb)
        uikg_emb=self.info_uikg_norm(uikg_user_emb)
        con_scores = F.linear(uikg_emb, con_nodes_features, self.info_output_con.bias)
        uikg_scores = F.linear(con_emb, uikg_nodes_features, self.info_output_uikg.bias)

        # info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.cuda().float()),dim=-1)*mask.cuda() # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        info_uikg_loss = torch.sum(
            self.info_uikg_loss(
                uikg_scores.to(DEVICE),
                uikg_label.to(DEVICE).float()
            ), dim=-1
        ) * mask.to(DEVICE)
        ##########################################################
        # info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.cuda().float()),dim=-1)*mask.cuda() # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        info_con_loss = torch.sum(
            self.info_con_loss(
                con_scores.to(DEVICE),
                con_label.to(DEVICE).float()
            ), dim=-1
        ) * mask.to(DEVICE)
        ##########################################################

        return torch.mean(info_uikg_loss), torch.mean(info_con_loss)

    def _rerank_with_similar_users(self, context_emb, node_features, scores,
                                   target_user_nodes, history_seen_nodes,
                                   history_disliked_nodes, labels, training):
        """Restrict movie ranking to neighbors' likes and apply KG preferences.

        Direct likes form the candidate set.  Target-user exclusions are read
        only from earlier dialogue turns, never from static KG user profiles.
        """
        user_features = node_features.index_select(0, self.uikg_user_nodes)
        similarities = torch.matmul(
            F.normalize(context_emb, dim=-1),
            F.normalize(user_features, dim=-1).transpose(0, 1))
        result = torch.full_like(scores, -1e9)

        #### Store top-k similar users for the current batch ####
        self.last_similar_user_ids = []
        self.last_similar_user_entity_ids = []
        self.last_similar_user_scores = []
        #########################################################

        for batch_index in range(scores.size(0)):
            target_node = int(target_user_nodes[batch_index].item())
            sim = similarities[batch_index].clone()
            # A target user is not their own "similar user"; this also avoids
            # leaking the target user's KG preferences into neighbor retrieval.
            if target_node >= 0:
                sim[self.uikg_user_nodes == target_node] = -float('inf')
            k = min(self.top_k_users, sim.numel())
            neighbor_scores, neighbor_idx = torch.topk(sim, k=k)

            # Save top-k similar user entity IDs and their cosine similarities
            # ---------------------------------------------------------
            # Save top-k similar users
            # ---------------------------------------------------------

            # UIKG entity IDs
            neighbor_user_entity_ids = self.uikg_user_nodes[neighbor_idx]

            # Convert UIKG entity IDs -> original ReDial user IDs
            neighbor_redial_user_ids = (
                neighbor_user_entity_ids - self.user_entity_offset
            )

            # Save original ReDial IDs
            self.last_similar_user_ids.append(
                neighbor_redial_user_ids.detach().cpu().tolist()
            )

            # Also save UIKG entity IDs for debugging/verification
            self.last_similar_user_entity_ids.append(
                neighbor_user_entity_ids.detach().cpu().tolist()
            )

            # Save cosine similarities
            self.last_similar_user_scores.append(
                neighbor_scores.detach().cpu().tolist()
            )
            #######################################################################

            candidates, support, genre_penalty = set(), {}, {}
            for similarity, index in zip(neighbor_scores.tolist(), neighbor_idx.tolist()):
                if similarity == float('-inf'):
                    continue
                neighbor = int(self.uikg_user_nodes[index].item())
                for movie in self.user_liked_movies.get(neighbor, []):
                    candidates.add(movie)
                    support[movie] = support.get(movie, 0.0) + self.neighbor_like_weight * similarity
                # Optional ablation: add a weak positive signal for movies
                # merely seen by neighbors, and a negative signal for movies
                # they disliked.  This never reads target-user KG preferences.
                if self.use_seen_dislike_neighbor_ablation:
                    for movie in self.neighbor_seen_movies.get(neighbor, []):
                        candidates.add(movie)
                        support[movie] = support.get(movie, 0.0) + self.neighbor_seen_weight * similarity
                    for movie in self.neighbor_disliked_movies.get(neighbor, []):
                        support[movie] = support.get(movie, 0.0) - self.neighbor_dislike_weight * similarity
                # Optional ablation: enable this flag to expand direct likes
                # with movies sharing a similar user's liked people (directors,
                # writers, producers, and starring cast) or liked genres.
                if self.enable_attribute_expansion:
                    for person in self.user_liked_people.get(neighbor, []):
                        for movie in self.person_to_movies.get(person, []):
                            candidates.add(movie)
                            support[movie] = support.get(movie, 0.0) + self.person_weight * similarity
                    for genre in self.user_liked_genres.get(neighbor, []):
                        for movie in self.genre_to_movies.get(genre, []):
                            candidates.add(movie)
                            support[movie] = support.get(movie, 0.0) + self.genre_weight * similarity
                    for genre in self.user_disliked_genres.get(neighbor, []):
                        for movie in self.genre_to_movies.get(genre, []):
                            genre_penalty[movie] = genre_penalty.get(movie, 0.0) + similarity

            retrieved_candidates = set(candidates)
            if self.use_seen_dislike_neighbor_ablation:
                candidates = {movie for movie in candidates
                              if support.get(movie, 0.0) > self.neighbor_support_threshold}

            # -1 is dataset padding.  Unlike the old static-profile variant,
            # these sets contain only mentions from dialogue turns < t.
            blocked = {int(movie) for movie in history_seen_nodes[batch_index].tolist()
                       if 0 <= int(movie) < scores.size(1)}
            blocked.update(int(movie) for movie in history_disliked_nodes[batch_index].tolist()
                           if 0 <= int(movie) < scores.size(1))
            candidates.difference_update(blocked)
            # Candidate retrieval is non-differentiable.  Ensure the observed
            # target remains trainable even if it was not retrieved.
            gold = int(labels[batch_index].item())
            if training and gold > 0 and gold not in blocked:
                candidates.add(gold)
            if not candidates: # Fallback: Use original retrieved candidates (before thresholding)
                # If weighted filtering is too strict, fall back first to the
                # retrieved neighbor set rather than every movie in the KG.
                candidates.update(retrieved_candidates)
                candidates.difference_update(blocked)
            if not candidates: # Fallback: Use ALL movies in the KG
                candidates.update(self.uikg_movie_nodes.tolist())
                candidates.difference_update(blocked)

            candidate_ids = torch.tensor(sorted(candidates), device=scores.device, dtype=torch.long)
            candidate_scores = scores[batch_index, candidate_ids]
            boost = torch.tensor([support.get(movie, 0.0) for movie in candidate_ids.tolist()],
                                 device=scores.device, dtype=scores.dtype)
            penalty = torch.tensor([genre_penalty.get(movie, 0.0) for movie in candidate_ids.tolist()],
                                   device=scores.device, dtype=scores.dtype)
            result[batch_index, candidate_ids] = (
                candidate_scores + self.neighbor_weight * boost
                - self.disliked_genre_weight * penalty)
        return result

    def forward(self, xs, ys, mask_ys, concept_mask, uikg_mask, seed_sets, labels, con_label, uikg_label, entity_vector, rec, target_user_nodes=None, history_seen_nodes=None, history_disliked_nodes=None, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        ###################################################################################
        set_seed(self.seed)
        ###################################################################################

        if test == False:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        #xxs = self.embeddings(xs)
        #mask=xs == self.pad_idx
        encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)

        # graph network
        uikg_nodes_features = self.uikg_RGCN(None, self.uikg_edge_idx, self.uikg_edge_type)
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        user_representation_list = []
        uikg_con_mask=[]
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                # user_representation_list.append(torch.zeros(self.dim).cuda()) # Original line based on torch 1.3.0 (avoid this)
                ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
                user_representation_list.append(torch.zeros(self.dim, device=DEVICE))
                ##########################################################
                uikg_con_mask.append(torch.zeros([1]))
                continue
            user_representation = uikg_nodes_features[seed_set]  # torch can reflect
            user_representation = self.self_attn_uikg(user_representation)
            user_representation_list.append(user_representation)
            uikg_con_mask.append(torch.ones([1]))

        uikg_user_emb=torch.stack(user_representation_list)
        uikg_con_mask=torch.stack(uikg_con_mask)

        graph_con_emb=con_nodes_features[concept_mask]
        con_emb_mask=concept_mask==self.concept_padding

        con_user_emb=graph_con_emb
        # con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda()) # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        con_user_emb, attention = self.self_attn(
            con_user_emb.to(DEVICE),
            con_emb_mask.to(DEVICE)
        )
        ##########################################################
        ######################## Change to context embedding ########################
        context_emb=self.user_norm(torch.cat([con_user_emb,uikg_user_emb],dim=-1))
        context_gate = F.sigmoid(self.gate_norm(context_emb))
        context_emb = context_gate * uikg_user_emb + (1 - context_gate) * con_user_emb
        entity_scores = F.linear(context_emb, uikg_nodes_features, self.output_en.bias)
        if target_user_nodes is None or history_seen_nodes is None or history_disliked_nodes is None:
                raise ValueError("Custom KG mode requires user IDs and dialogue-history exclusions from dataset.py")
        entity_scores = self._rerank_with_similar_users(
            context_emb, uikg_nodes_features, entity_scores, target_user_nodes,
            history_seen_nodes, history_disliked_nodes, labels, training=not test)
        # top_user_indices, top_user_scores = self.retrieve_similar_users(
        #     context_emb, self.user_embeddings, self.top_k_users
        # )
        # candidate_movie_ids = self.get_liked_movie_candidates(top_user_indices) ## Not definied yet
        # candidate_movie_embs = uikg_nodes_features[candidate_movie_ids]
        # self.context_movie_match = nn.Bilinear(
        #     self.dim, self.dim, 1, bias=False
        # )

        # candidate_scores = self.context_movie_match(
        #     context_emb.unsqueeze(1).expand_as(candidate_movie_embs),
        #     candidate_movie_embs
        # ).squeeze(-1)

        # # context_proj = self.context_to_movie(context_emb)
        # # candidate_scores = torch.bmm(
        # #     candidate_movie_embs,
        # #     context_proj.unsqueeze(-1)
        # # ).squeeze(-1)
        #############################################################################
        #entity_scores = scores_db * gate + scores_con * (1 - gate)
        #entity_scores=(scores_db+scores_con)/2

        #mask loss
        #m_emb=db_nodes_features[labels.cuda()] # Original line based on torch 1.3.0 (avoid this)
        #mask_mask=concept_mask!=self.concept_padding
        mask_loss=0#self.mask_predict_loss(m_emb, attention, xs, mask_mask.cuda(),rec.float()) # Original line based on torch 1.3.0 (avoid this)

        info_uikg_loss, info_con_loss=self.infomax_loss(con_nodes_features,uikg_nodes_features,con_user_emb,uikg_user_emb,con_label,uikg_label,uikg_con_mask)

        #entity_scores = F.softmax(entity_scores.cuda(), dim=-1).cuda() # Original line based on torch 1.3.0 (avoid this)

        # rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda()) # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        rec_loss = self.criterion(
            entity_scores.squeeze(1).squeeze(1).float().to(DEVICE),  # Ensures float32 on correct device
            labels.to(DEVICE, dtype=torch.long)  # Ensures long dtype for labels
        )
        # rec_loss = self.criterion(
        #             candidate_scores.squeeze(1).squeeze(1).float().to(DEVICE),  # Ensures float32 on correct device
        #             labels.to(DEVICE, dtype=torch.long)  # Ensures long dtype for labels
        #         )
        ##########################################################
        #rec_loss=self.klloss(entity_scores.squeeze(1).squeeze(1).float(), labels.float().cuda()) # Original line based on torch 1.3.0 (avoid this)
        # rec_loss = torch.sum(rec_loss*rec.float().cuda()) # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        rec_loss = torch.sum(rec_loss * rec.to(DEVICE).float())
        ##########################################################

        self.context_rep=context_emb

        #generation---------------------------------------------------------------------------------------------------
        con_nodes_features4gen=con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
        con_emb4gen = con_nodes_features4gen[concept_mask]
        con_mask4gen = concept_mask != self.concept_padding
        #kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
        # kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda()) # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        kg_encoding = (
            self.kg_norm(con_emb4gen.to(DEVICE)),  # Normalize then move to device
            con_mask4gen.to(DEVICE)                # Move mask to same device
        )
        ##########################################################

        uikg_emb4gen=uikg_nodes_features[entity_vector] #batch*50*dim
        uikg_mask4gen=entity_vector!=0
        #db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
        # db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.cuda()) # Original line based on torch 1.3.0 (avoid this)
        ########################################################## Modernized version based on torch 2.3.0 (recommended) added by T.S
        uikg_encoding = (
            self.uikg_norm(uikg_emb4gen.to(DEVICE)),  # Move input to device BEFORE normalization
            uikg_mask4gen.to(DEVICE)                # Move mask to device
        )
        ##########################################################

        if test == False:
            # use teacher forcing
            scores, preds = self.decode_forced(encoder_states, kg_encoding, uikg_encoding, con_user_emb, uikg_user_emb, mask_ys)
            gen_loss = torch.mean(self.compute_loss(scores, mask_ys))

        else:
            scores, preds = self.decode_greedy(
                encoder_states, kg_encoding, uikg_encoding, con_user_emb, uikg_user_emb,
                bsz,
                maxlen or self.longest_label
            )
            gen_loss = None

        return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, info_con_loss
        # return scores, preds, candidate_scores, rec_loss, gen_loss, mask_loss, info_uikg_loss, info_con_loss

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    # def compute_loss(self, output, scores):
    #     score_view = scores.view(-1)
    #     output_view = output.view(-1, output.size(-1))
    #     loss = self.criterion(output_view.cuda(), score_view.cuda()) # Original line based on torch 1.3.0 (avoid this)
    #     return loss
    ######################################################### Change function based on torch 2.3.0+cu118 by T.S
    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        
        # Ensure correct data types and device placement
        score_view = score_view.to(device=DEVICE, dtype=torch.long)  # Critical: targets must be long
        output_view = output_view.to(DEVICE)  # Uses global DEVICE variable
        
        loss = self.criterion(output_view, score_view)
        return loss
    #########################################################

    def save_model(self):
        ###################################################################################
        set_seed(self.seed)
        ###################################################################################

        """Save model parameters to disk""" # this line added by T.S
        ################################################# Added by T.S
        # Create the directory if it doesn't exist
        os.makedirs(self.path_KG + 'saved_model', exist_ok=True)
        ########################################################
        torch.save(self.state_dict(), self.path_KG + 'saved_model/net_parameter1.pkl')

    def load_model(self):
        ###################################################################################
        set_seed(self.seed)
        ###################################################################################

        self.load_state_dict(torch.load(self.path_KG + 'saved_model/net_parameter1.pkl'))

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.context_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output



# Following codes just add by myself to check somthings and these are not belong to original code and has not any relation with
# algorithm of paper
# 
# kg = pkl.load(
#             open("data/subkg.pkl", "rb")
#         )

# print(type(kg))
# print(len(kg))
# print(list(kg.items())[0:5])
# print(0 in kg.keys())
# print('-'*100)

# e2eid = pkl.load(
#             open("data/entity2entityId.pkl", "rb")
#         )

# print(type(e2eid))
# print(len(e2eid))
# print(list(e2eid.items())[0:5])
# print(list(e2eid.keys())[0:5])
# print(list(e2eid.values())[-1])
# print('-'*100)

# id2e = pkl.load(
#             open("data/id2entity.pkl", "rb")
#         )

# print(type(id2e))
# print(len(id2e))
# print(list(id2e.items())[0:5])
# print(list(id2e.keys())[0:5])
# print(list(id2e.keys())[-1])
# print(list(id2e.values())[-1])
# print('-'*100)

# mids = pkl.load(
#             open("data/movie_ids.pkl", "rb")
#         )

# print(type(mids))
# print(len(mids))
# print(mids[0:5])
# print(mids[-1])
# print('-'*100)

# text_dict = pkl.load(
#             open("data/text_dict.pkl", "rb")
#         )

# print(type(text_dict))
# print(len(text_dict))
# print(list(text_dict.items())[0:5])
# print(list(text_dict.keys())[0:5])
# print(list(text_dict.keys())[-1])
# print(list(text_dict.values())[-1])