from itertools import chain

from torch import nn
from masked_cross_entropy import *
from utils import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import clamp_grad
import copy
from itertools import chain, repeat
from torch.nn import functional as F

from modules.BinaryTreeBasedModule import BinaryTreeBasedModule

USE_CUDA = torch.cuda.is_available()

PAD_token = 0
SOS_token = 1
EOS_token = 2
x1_token = 3
x2_token = 4
x3_token = 5
x4_token = 6

all_entities = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
available_src_vars = ['x1', 'x2', 'x3', 'x4']


class BottomAbstrator(nn.Module):
    # To make bottom abstractions such as 'M0' and 'executive produce'
    def __init__(self, alignment_idx):
        super().__init__()
        self.alignment_idx = alignment_idx

    def forward(self, x):
        bottom_idx = []
        for position, token in enumerate(x):
            if token.item() in self.alignment_idx:
                bottom_idx.append(position)
            else:
                continue

        return bottom_idx


class BottomClassifier(nn.Module):
    # To classify bottom abstractions
    def __init__(self, output_lang, alignments_idx):
        super().__init__()
        self.output_lang = output_lang
        self.alignments_idx = alignments_idx

    def forward(self, x, bottom_idx):
        idx2output_token = []
        for idx in bottom_idx:
            input_idx = x[idx].item()
            output_idx = self.alignments_idx[input_idx]
            assert len(output_idx) == 1
            output_idx = output_idx[0]
            output_token = self.output_lang.index2word[output_idx]
            idx2output_token.append([idx, output_token])

        return idx2output_token


class DepTreeComposer(BinaryTreeBasedModule):
    def __init__(self, input_dim, hidden_dim, composer_trans_hidden, dropout_prob=0.1, 
                 leaf_transformation="bi_lstm_transformation", relaxed=False, tau_weights=None, 
                 straight_through=False):
        super().__init__(input_dim, hidden_dim, leaf_transformation, 
                         composer_trans_hidden, dropout_prob)

        self.arc_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.hidden_dim = hidden_dim
        self.tau_weights = tau_weights
        self.straight_through = straight_through
        self.relaxed = relaxed
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.arc_scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)


    def batch_forward(self, x_emb, bottom_all, word_mask):
        """
        x_emb      : (S, L_max, D)   — mega-batch (B×N sentences)
        bottom_all : (S, B_max) int  — bottom indices padded with −1
        word_mask  : (S, L_max) float — 1 for real tokens
        """
        S, L_max, D = x_emb.shape
        #printf"S: {S}, L_max: {L_max}, D: {D}")
        device      = x_emb.device

        # 1) Leaf encoding (unchanged)
        mask_leaf   = bottom_all >= 0                            # (S, B_max)
        #printf"mask_leaf shape: {mask_leaf.shape}")
        #printf"mask_leaf: {mask_leaf}")
        leaf_pos   = bottom_all.clone()
        leaf_pos[~mask_leaf] = 0
        #printf"leaf_pos shape: {leaf_pos.shape}")
        #printf"leaf_pos: {leaf_pos}")
        max_B       = mask_leaf.sum(dim=1).max().item()          # largest #leaves in batch
        #printf"max_B: {max_B}")

        # get full leaf states
        h_all, c_all = self._transform_leafs(x_emb, word_mask)   # (S, L_max, D)
        #printf"h_all shape: {h_all.shape}")
        #printf"c_all shape: {c_all.shape}")

        # gather out only the "bottom" positions into B_max slots
        h0 = h_all.gather(1, leaf_pos.unsqueeze(-1).expand(-1,-1,D))
        c0 = c_all.gather(1, leaf_pos.unsqueeze(-1).expand(-1,-1,D))
        #printf"h0 shape: {h0.shape}")
        #printf"c0 shape: {c0.shape}")

        # zero out padded leaves
        h0 = h0.masked_fill(~mask_leaf.unsqueeze(-1), 0.0)
        c0 = c0.masked_fill(~mask_leaf.unsqueeze(-1), 0.0)
        #printf"h0 after masking shape: {h0.shape}")
        #printf"c0 after masking shape: {c0.shape}")

        # init working tensors
        hidden      = h0.clone()                                 # (S, B_max, D)
        cell        = c0.clone()
        mask        = mask_leaf.float()                          # (S, B_max)
        idx_lookup  = bottom_all.clone()                         # (S, B_max)

        # bookkeeping
        log_p       = []
        ent         = []
        edges_all   = [[] for _ in range(S)]
        idx2ctx     = [ {} for _ in range(S) ]

        # pre-fill idx2ctx with initial leaf contexts
        for s in range(S):
            for j in range(mask_leaf.size(1)):
                tok = idx_lookup[s, j].item()
                if tok >= 0:
                    idx2ctx[s][str(tok)] = hidden[s, j].clone()
        #printf"each idx2ctx[s] has {len(idx2ctx[0])} entries")


        # --- NEW: per-sample merge loop --------------------------------
        # how many merges each sample needs = (#leaves - 1)
        leaf_counts   = mask_leaf.sum(dim=1).to(torch.int64)    # (S,)
        #printf"leaf_counts shape: {leaf_counts.shape}")
        merge_counts  = torch.zeros_like(leaf_counts)           # (S,)
        #printf"merge_counts shape: {merge_counts.shape}")
        active        = merge_counts < (leaf_counts - 1)        # (S,) bool
        #printf"active shape: {active.shape}")

        active_masks = []
        while active.any():
            active_masks.append(active.clone())
            # one batch step
            sel_h, sel_d, lp_batch, ne_batch, hidden, cell, mask = \
                self._make_step_batch(hidden, cell, mask)
            #printf"sel_h shape: {sel_h.shape}, sel_d shape: {sel_d.shape}")
            #printf"lp_batch shape: {lp_batch.shape}, ne_batch shape: {ne_batch.shape}")
            
            # collect only for still-active samples
            log_p.append(lp_batch[active])
            ent.append(ne_batch[active])
            #printf"log_p: {log_p[-1]}")
            #printf"ent: {ent[-1]}")

            # update per-sample books
            #printf"active nonzero: {active.nonzero(as_tuple=True)}")
            for s in active.nonzero(as_tuple=True)[0].tolist():
                #printf"Processing sample {s} with sel_h: {sel_h[s]}, sel_d: {sel_d[s]}")
                h_j = sel_h[s].item()
                #printf"h_j: {h_j}")
                d_j = sel_d[s].item()
                #printf"d_j: {d_j}")
                h_tok = idx_lookup[s, h_j].item()
                #printf"h_tok: {h_tok}")
                d_tok = idx_lookup[s, d_j].item()
                #printf"d_tok: {d_tok}")
                # only if both are real leaves
                if h_tok >= 0 and d_tok >= 0:
                    edges_all[s].append((h_tok, d_tok))
                    #printf"Appending edge ({h_tok}, {d_tok}) to edges_all[{s}]")
                    idx2ctx[s][str(h_tok)] = hidden[s, h_j].clone()
                    #printf"Updating idx2ctx[{s}][{h_tok}] with hidden[{s}, {h_j}]")
                # mark that position as merged
                idx_lookup[s, d_j] = -1
                merge_counts[s]    += 1
                #printf"Updated idx_lookup[{s}, {d_j}] to -1 and incremented merge_counts[{s}] to {merge_counts[s]}")
            # recompute who still needs more merges
            active = merge_counts < (leaf_counts - 1)

        # 3) package RL info back into per-sample tensors
        total_steps = len(log_p)
        stacked_lp = torch.zeros(total_steps, S, device=device)
        stacked_ne = torch.zeros_like(stacked_lp)

        for t, (lp_t, ne_t, m) in enumerate(zip(log_p, ent, active_masks)):
            # m is the Boolean mask that was stored just before _make_step_batch
            stacked_lp[t, m] = lp_t           # (len(lp_t) == m.sum())
            stacked_ne[t, m] = ne_t

        #printf"stacked_lp shape: {stacked_lp.shape}")
        #printf"stacked_lp: {stacked_lp}")
        #printf"stacked_ne shape: {stacked_ne.shape}")
        #printf"stacked_ne: {stacked_ne}")

        rl_info = {
            "log_prob":           stacked_lp.sum(0),   # (S,)
            "normalized_entropy": stacked_ne.mean(0),  # (S,)
            "reward":             torch.zeros(S, device=device)
        }
        return edges_all, idx2ctx, rl_info


    def _make_step_batch(self, hidden, cell, mask):
        """
        hidden, cell : (S, L_max, D)
        mask         : (S, L_max)   1 = node still unattached
        returns:
            sel_h, sel_d          : (S,)      int32 head / dep indices
            logp, entropy         : (S,)      RL scalars
            hidden, cell, mask    : updated tensors (same shapes)
        """
        S, L, D   = hidden.shape
        device    = hidden.device

        # 1. Build all head×dep representations in one shot
        h_h = hidden.unsqueeze(2).expand(-1, -1, L, -1)   # (S,L,L,D)
        h_d = hidden.unsqueeze(1).expand(-1, L, -1, -1)
        c_h = cell  .unsqueeze(2).expand_as(h_h)
        c_d = cell  .unsqueeze(1).expand_as(c_h)
        #printf"h_h shape: {h_h.shape}")
        #printf"h_d shape: {h_d.shape}")
        #printf"c_h shape: {c_h.shape}")
        #printf"c_d shape: {c_d.shape}")


        new_h, new_c = self.tree_lstm_cell(                      # (S,L,L,D)
            h_h.reshape(-1, D), c_h.reshape(-1, D),
            h_d.reshape(-1, D), c_d.reshape(-1, D)
        )
        #printf"new_h shape: {new_h.shape}")
        #printf"new_c shape: {new_c.shape}")

        new_h = new_h.view(S, L, L, D)
        new_c = new_c.view_as(new_h)
        #printf"new_h after view shape: {new_h.shape}")
        #printf"new_c after view shape: {new_c.shape}")

        # 2. score every arc once
        arc_feat = torch.cat([h_h, h_d], dim=-1)                 # (S,L,L,2D)
        score    = self.arc_scorer(arc_feat).squeeze(-1)         # (S,L,L)
        #printf"arc_feat shape: {arc_feat.shape}")
        #printf"score shape: {score.shape}")
        

        # 3. mask: no self-loop; dep must still be free
        dep_free   = mask.bool().unsqueeze(1)
        arc_mask   = torch.ones_like(score)
        arc_mask   *= dep_free
        arc_mask.diagonal(dim1=1, dim2=2).fill_(0)
        score_masked = score.masked_fill(arc_mask == 0, torch.finfo(score.dtype).min)
        flat_mask   = arc_mask.view(S, -1).float()
        flat_score = score_masked.view(S, -1)


        flat_score = torch.nan_to_num(flat_score, nan=0.0, posinf=50.0, neginf=-50.0)
        cat = Categorical(flat_score, flat_mask)                 # utils.Categorical
        actions, _ = self._sample_action(cat, flat_mask)         # (S,L²) one-hot
        sel_idx    = actions.argmax(dim=-1)                      # (S,) 0…L²−1
        sel_h, sel_d = torch.div(sel_idx, L, rounding_mode='floor'), sel_idx % L

        # 5. update hidden / cell only at sel_h — but safely, via clone + indexed write
        batch = torch.arange(S, device=device)
        # gather the new parent states for each example
        updates_h = new_h[batch, sel_h, sel_d]   # (S, D)
        updates_c = new_c[batch, sel_h, sel_d]   # (S, D)

        # clone so we don’t corrupt any autograd metadata on the original tensors
        hidden_new = hidden.clone()              # (S, L, D)
        cell_new   = cell.clone()                # (S, L, D)

        # indexed write into the clones
        hidden_new[batch, sel_h] = updates_h     # safe inplace on the clone
        cell_new  [batch, sel_h] = updates_c

        # rebind
        hidden, cell = hidden_new, cell_new

        # mark dependent as attached
        mask[batch, sel_d] = 0.0

        return sel_h, sel_d, cat.log_prob(sel_idx), cat.normalized_entropy, hidden, cell, mask



    def _sample_action(self, cat_distr, mask):
        noise = None
        if self.training:
            if self.relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = self.tau_weights[0] + self.tau_weights[1].exp() * torch.log(N + 1) + self.tau_weights[2].exp() * N
                actions, noise = cat_distr.rsample(temperature=tau, gumbel_noise=noise)
                if self.straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise
    

class Solver(nn.Module):
    """ Solver class for the COGS model.
    
    This class is responsible for generating the final output of the model.
    It takes the output of the BottomUpTreeComposer and generates the final output using the semantic classifier.
    """
    # To compose bottom abstractions with rules
    def __init__(self, hidden_dim, output_lang, entity_list=[], caus_predicate_list=[], 
                 unac_predicate_list=[], relaxed=False, tau_weights=None, straight_through=False, gumbel_noise=None,
                eval_actions=None, eval_sr_actions=None, eval_swr_actions=None, debug_info=None):
        super().__init__()

        # 0: in
        # 1: on
        # 2: beside
        self.semantic_E_E = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # in, on, beside
        )

        # 0: agent
        # 1: theme
        # 2: recipient
        self.semantic_P_E = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # agent, theme, recipient
        )


        # 0: ccomp
        # 1: xcomp
        self.semantic_P_P = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # ccomp, xcomp
        )
        
        self.hidden_dim = hidden_dim
        self.output_lang = output_lang

        self.entity_list = entity_list
        self.caus_predicate_list = caus_predicate_list
        self.unac_predicate_list = unac_predicate_list
        self.predicate_list = caus_predicate_list + unac_predicate_list

        self.relaxed = relaxed
        self.tau_weights = tau_weights
        self.straight_through = straight_through
        self.gumbel_noise = gumbel_noise
        self.eval_actions = eval_actions
        self.eval_sr_actions = eval_sr_actions
        self.eval_swr_actions = eval_swr_actions
        self.debug_info = debug_info

    def sort_bottom_up(self, pairs):
        """ Sort dependent pairs in a bottom-up manner.
        
        This is so the tree can be traversed bottom-up and no parent is visited before its children.
        """
        # Build parent mapping and collect children
        parent_map = {dep: head for head, dep in pairs if head != dep}
        def get_depth(node):
            depth = 0
            while node in parent_map and node != parent_map[node]:
                node = parent_map[node]
                depth += 1
            return depth

        # Sort pairs by depth of the dependent node
        pairs_no_root = [pair for pair in pairs if pair[0] != pair[1]]  # exclude root self-pair
        return sorted(pairs_no_root, key=lambda pair: get_depth(pair[1]), reverse=True)

    def batch_forward(self,
                    pair_list,          # list length = N
                    edges_list,         # list length = N
                    idx2ctx_list,       # list length = N
                    outtok_list):       # list length = N
        # 1. shape sanity-check ------------------------------------------------
        assert len(pair_list) == len(edges_list) == len(idx2ctx_list) == len(outtok_list), \
            "All arguments must have the same length (= batch-size N)"

        sem_edges_list, rl_parts, dbg = [], [], []

        # 2. run the per-sample solver ----------------------------------------
        for n in range(len(edges_list)):
            sem_edges, rl, d = self.forward(
                pair_list[n],            # ← one sentence
                edges_list[n],
                idx2ctx_list[n],
                outtok_list[n]           # ← one [idx, token] list
            )
            sem_edges_list.append(sem_edges)
            rl_parts.append(rl)
            dbg.append(d)

        # 3. stack the scalar RL tensors --------------------------------------
        rl_info_stacked = {
            k: torch.cat([p[k] for p in rl_parts], dim=0)
            for k in rl_parts[0]          # 'log_prob', 'normalized_entropy', …
        }

        # keep a slot for the reward the caller fills later
        device = rl_info_stacked["log_prob"].device     # safe – this *is* a tensor
        rl_info_stacked["reward"] = torch.zeros(len(edges_list), device=device)

        return sem_edges_list, rl_info_stacked, dbg
    

    def forward(self, pair, head_dep_idx, idx2contextual, idx2output_token):
        debug_info = {}
        entropy_parts  = []
        logprob_parts  = []


        # returns either E or P instances depending of whether the token is on the entity or predicate list
        # idx2output_token is a list of pairs like [ 1, "cake"], [3, "cook"]]
        # idx2semantic is a dictionary of pairs like {"1": E("cake"), "3": P("cook")}
        idx2semantic = self.init_semantic_class(idx2output_token)

        debug_info["idx2semantic"] = idx2semantic
        debug_info["head_dep_idx"] = head_dep_idx
        debug_info["idx2output_token"] = idx2output_token

        semantic_edges = []

        semantic_normalized_entropy = []
        semantic_log_prob = []

        # Example tree: [(8, 2), (8, 3), (8, 5), (1, 8)]
        # Example idx2semantic: {"1": E("cake"), "3": P("cook")}

        for head, dep in head_dep_idx:
            # get semantic class of the head and dependent
            assert str(head) in idx2semantic, f"Head {head} not found in idx2semantic: {idx2semantic}. \nidx2output_token{idx2output_token}\nhead_dep_idx: {head_dep_idx}"
            head_sem = idx2semantic[str(head)]

            assert str(dep) in idx2semantic, f"Dependent {dep} not found in idx2semantic: {idx2semantic}. \nidx2output_token{idx2output_token}\nhead_dep_idx: {head_dep_idx}"
            dep_sem = idx2semantic[str(dep)]

            head_contextual = idx2contextual[str(head)]
            dep_contextual = idx2contextual[str(dep)]

            semantic_input = torch.cat([head_contextual, dep_contextual], dim=-1)
            

            cat_distr, _, actions_i, chosen_rel = self.semantic_merge(head_sem, dep_sem, semantic_input) 

            semantic_edges.append((str(head), str(dep), chosen_rel))


            if cat_distr is not None:
                # collect for RL
                entropy_parts.append(cat_distr.normalized_entropy)
                action_idx = actions_i.argmax(dim=-1)              # shape: (), a 0-D tensor
                logprob_parts.append(cat_distr.log_prob(action_idx))



        debug_info["idx2semantic"] = idx2semantic
        debug_info["semantic_edges"] = semantic_edges

        # calculate the normalized entropy and log probability of the actions taken for RL
        if len(entropy_parts) == 0:              # no stochastic choice in sentence
            try:
                device = next(iter(idx2contextual.values())).device
            except StopIteration:                     # idx2contextual is empty
                device = torch.zeros(1).device        # defaults to current default device

            normalized_entropy = torch.tensor(0.0, device=device)
            semantic_log_prob  = torch.tensor(0.0, device=device)
        else:
            normalized_entropy = sum(entropy_parts) / len(entropy_parts)
            semantic_log_prob  = sum(logprob_parts)


        rl_info = {
            "normalized_entropy": normalized_entropy.unsqueeze(0),  # shape (1,)
            "log_prob":            semantic_log_prob.unsqueeze(0),   # shape (1,)
            "reward":              torch.zeros(1, device=normalized_entropy.device)
        }
                   

        return semantic_edges, rl_info, debug_info

    def init_semantic_class(self, idx2output_token):
        idx2semantic = {}
        for idx in idx2output_token:
            idx_position = idx[0]
            output_token = idx[1]

            assert output_token in self.entity_list + self.predicate_list, f"Output token {output_token} not in entity or predicate list. \nidx was: {idx}\nidx2output_token was: {idx2output_token}"

            if output_token in self.entity_list:
                idx_semantic = "E"
            else:
                idx_semantic = "P"

            idx2semantic[str(idx_position)] = idx_semantic

        return idx2semantic

    def semantic_merge(self, head_sem, dep_sem, semantic_input):
        """ Compose the two children into a parent semantic class. 
        
        This is done by selecting a semantic operation from the semantic classifier.
        """
        # E2E classifier
        if head_sem == 'E' and dep_sem == 'E':
            semantic_score = self.semantic_E_E(semantic_input)
            semantic_score = torch.nan_to_num(semantic_score, nan=0.0, posinf=50.0, neginf=-50.0)

            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, #
                                                        semantic_mask, 
                                                        self.relaxed, 
                                                        self.tau_weights,
                                                        self.straight_through,
                                                        self.gumbel_noise)
            # 0: in, 1: on, 2: beside
            rel_dict = {0: 'in', 1: 'on', 2: 'beside'}
            chosen_rel = rel_dict[actions.argmax().item()]

        # P2P classifier
        elif head_sem== 'P' and dep_sem== 'P':
            semantic_score = self.semantic_P_P(semantic_input)
            semantic_score = torch.nan_to_num(semantic_score, nan=0.0, posinf=50.0, neginf=-50.0)

            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, #
                                                        semantic_mask, 
                                                        self.relaxed, 
                                                        self.tau_weights,
                                                        self.straight_through,
                                                        self.gumbel_noise)
            # 0: ccomp, 1: xcomp
            rel_dict = {0: 'ccomp', 1: 'xcomp'}
            chosen_rel = rel_dict[actions.argmax().item()]

        # P2E classifier
        elif head_sem== 'P' and dep_sem == 'E':
            semantic_score = self.semantic_P_E(semantic_input)
            semantic_score = torch.nan_to_num(semantic_score, nan=0.0, posinf=50.0, neginf=-50.0)

            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, #
                                                    semantic_mask, 
                                                    self.relaxed, 
                                                    self.tau_weights,
                                                    self.straight_through,
                                                    self.gumbel_noise)
            # 0: agent, 1: theme, 2: recipient
            rel_dict = {0: 'agent', 1: 'theme', 2: 'recipient'}
            chosen_rel = rel_dict[actions.argmax().item()]
            
        # E2P classifier
        elif head_sem == 'E' and dep_sem== 'P':
            # No classifier for this case
            # we know its a relcl
            chosen_rel = 'relcl'
            cat_distr, gumbel_noise, actions = None, None, None

        else:
            raise ValueError(f"Invalid semantic class: {head_sem}, {dep_sem}")

        return cat_distr, gumbel_noise, actions, chosen_rel


    def _sample_action(self, cat_distr, mask, relaxed, tau_weights, straight_through, gumbel_noise):
        if self.training:
            if relaxed:
                N = mask.sum(dim=-1, keepdim=True)
                tau = tau_weights[0] + tau_weights[1].exp() * torch.log(N + 1) + tau_weights[2].exp() * N
                actions, gumbel_noise = cat_distr.rsample(temperature=tau, gumbel_noise=gumbel_noise)
                if straight_through:
                    actions_hard = torch.zeros_like(actions)
                    actions_hard.scatter_(-1, actions.argmax(dim=-1, keepdim=True), 1.0)
                    actions = (actions_hard - actions).detach() + actions
                actions = clamp_grad(actions, -0.5, 0.5)
            else:
                actions, gumbel_noise = cat_distr.rsample(gumbel_noise=gumbel_noise)
        else:
            actions = torch.zeros_like(cat_distr.probs)
            actions.scatter_(-1, torch.argmax(cat_distr.probs, dim=-1, keepdim=True), 1.0)
            gumbel_noise = None
        return actions, gumbel_noise


class HRLModel(nn.Module):
    def __init__(self, vocab_size, word_dim, hidden_dim, label_dim,
                 composer_trans_hidden=None,
                 var_normalization=False,
                 input_lang=None, output_lang=None,
                 alignments_idx={}, entity_list=[], caus_predicate_list=[], unac_predicate_list=[]):
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.label_dim = label_dim
        self.alignments_idx = alignments_idx
        self.entity_list = entity_list
        self.caus_predicate_list = caus_predicate_list
        self.unac_predicate_list = unac_predicate_list
        self.embd_parser = nn.Embedding(vocab_size, word_dim)
        #self.position_embedding = nn.Embedding(100, word_dim)

        self.abstractor = BottomAbstrator(alignments_idx)
        self.classifier = BottomClassifier(output_lang, alignments_idx)

        self.composer = DepTreeComposer(input_dim=word_dim, hidden_dim=hidden_dim, 
                                        composer_trans_hidden=composer_trans_hidden)
        self.solver = Solver(hidden_dim, output_lang,
                             entity_list=entity_list,
                             caus_predicate_list=caus_predicate_list,
                             unac_predicate_list=unac_predicate_list)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.reset_parameters()
        self.is_test = False

        self.case = None

    def get_high_parameters(self):
        return list(self.embd_parser.parameters()) + list(self.composer.parameters())

    def get_low_parameters(self):
        return list(self.solver.parameters())

    def forward(self, pair, x, sample_num, is_test=False, epoch=None):
        self.is_test = is_test
        batch_forward_info, pred_chain, label_chain, state = self._forward(
            pair, x, sample_num, epoch, is_test)
        return batch_forward_info, pred_chain, label_chain, state

    def _forward(self, pair_batch, x_batch, sample_num, epoch, is_test):
        """
        pair_batch : list length B
        x_batch    : LongTensor (B, L)
        """
        #printf"pair_batch: {pair_batch}")
        #printf"x_batch: {x_batch}")
        state = {}

        B, L = x_batch.shape
        device = x_batch.device

        #printf"HRLModel forward: B={B}, L={L}")
        word_mask = (x_batch != PAD_token).float()   # (B, L)
        #printf"word_mask shape: {word_mask.shape}")
        #printf"word_mask: {word_mask}")

        # 1. bottom abstraction for every sentence
        bottom_all, idx2outtok_all = [], []
        for b in range(B):
            bottom_idx = self.abstractor(x_batch[b])
            bottom_all.append(bottom_idx)
            idx2outtok_all.append(self.classifier(x_batch[b], bottom_idx))
        #printf"bottom_all: {bottom_all}")
        #printf"idx2outtok_all: {idx2outtok_all}")

        # 2. embeddings once → (B, L, D)
        pos = torch.arange(L, device=device)
        x_emb = self.embd_parser(x_batch)# + self.position_embedding(pos)

        #printf"x_emb shape: {x_emb.shape}")

        # 3. replicate the *sample* dim, then flatten (B*N, L, D)
        N = sample_num
        x_emb = x_emb.unsqueeze(1).repeat(1, N, 1, 1).view(B*N, L, -1)        # (B*N,L,D)
        #printf"x_emb after repeat: {x_emb.shape}")

        # same for bottom indices
        max_B = max(len(b) for b in bottom_all)
        #printf"max_B: {max_B}")
        bottom_pad = torch.full((B, max_B), -1, device=device, dtype=torch.long)
        #printf"bottom_pad shape: {bottom_pad.shape}")
        #printf"bottom_pad: {bottom_pad}")
        for b, idx in enumerate(bottom_all):
            bottom_pad[b, :len(idx)] = torch.tensor(idx, device=device, dtype=torch.long)
        #printf"bottom_pad shape after filling: {bottom_pad.shape}")
        #printf"bottom_pad after filling: {bottom_pad}")
        
        bottom = bottom_pad.repeat_interleave(N, dim=0)
        #printf"bottom shape after repeat: {bottom.shape}")
        #printf"bottom after repeat: {bottom}")

        # 4. composer / solver ONCE on the merged batch
        word_mask_rep = word_mask.repeat_interleave(N, 0)  # (B*N, L)
        #printf"word_mask_rep shape: {word_mask_rep.shape}")
        #printf"word_mask_rep: {word_mask_rep}")
        edges, idx2ctx, comp_rl = self.composer.batch_forward(x_emb, bottom, word_mask_rep)
        #printf"edges {edges}")
        #printf"comp_rl {comp_rl}")

        idx2outtok_rep = list(chain.from_iterable(repeat(m, N) for m in idx2outtok_all))
        pair_rep = list(chain.from_iterable(repeat(p, N) for p in pair_batch))
        #printf"idx2outtok_rep shape: {len(idx2outtok_rep)}")
        #printf"pair_rep shape: {len(pair_rep)}")
        
        idx2ctx_detached = [
            {k: v.detach() for k, v in ctx.items()}
            for ctx in idx2ctx
        ]
        sem_edges, sol_rl, _ = self.solver.batch_forward(
            pair_rep, edges, idx2ctx_detached, idx2outtok_rep
        )

        #printf"sem_edges: {sem_edges}")
        #printf"sol_rl: {sol_rl}")
        
        # 5. reward + bookkeeping (vectorised)
        N = sample_num                     # already defined
        
        gold_edges_base = [self.process_gold(p[1])[0] for p in pair_batch]   # length B
        #printf"gold_edges_base: {gold_edges_base}")
        tok_maps_base   = [{str(i): w for i, w in m}
                        for m in idx2outtok_all]                          # length B
        #printf"tok_maps_base: {tok_maps_base}")

        # replicate sentence-level structures so they line up with sem_edges (B*N)
        gold_edges_rep = [gold_edges_base[b] for b in range(B) for _ in range(N)]
        tok_maps_rep   = [tok_maps_base[b]   for b in range(B) for _ in range(N)]
        #printf"gold_edges_rep: {gold_edges_rep}")
        #printf"tok_maps_rep: {tok_maps_rep}")

        assert len(gold_edges_rep) == B*N
        assert len(tok_maps_rep)   == B*N     # sanity

        rewards         = torch.zeros(B*N, 2, device=device)
        pred_edges_all  = []

        for i in range(B*N):
            m    = tok_maps_rep[i]
            gold = gold_edges_rep[i]
            pred = [[m[e[0]], m[e[1]], e[2]] for e in sem_edges[i]]

            c_r, s_r = self.get_reward(pred, gold)

            comp_rl["reward"][i] = c_r
            sol_rl["reward"][i]  = s_r
            pred_edges_all.append(pred)
            rewards[i] = torch.tensor([c_r, s_r], device=device)
            #printf"Processing sentence {i}:")
            #printf"  Predicted edges: {pred}")
            #printf"  Gold edges: {gold}")
            #printf"composer reward: {c_r}, solver reward: {s_r}")

        # 6. reshape back so training loop’s `flat = …` still works
        comp_split = [{k: v[i].unsqueeze(0) for k, v in comp_rl.items()}
                    for i in range(B * N)]
        #printf"comp_split: {comp_split}")
        sol_split  = [{k: v[i].unsqueeze(0) for k, v in sol_rl.items()}
                    for i in range(B * N)]
        #printf"sol_split: {sol_split}")
        batch_forward = [[comp_split[i * N + n], sol_split[i * N + n]]
                        for i in range(B) for n in range(N)]
        #printf"batch_forward: {batch_forward}")
        #printf"pred_edges_all: {pred_edges_all}")

        pred_per_sentence  = [pred_edges_all[b * N]   for b in range(B)]
        gold_per_sentence  = [gold_edges_base[b]      for b in range(B)]
        comp_reward_sent = [comp_rl["reward"][b * N].item() for b in range(B)]
        sol_reward_sent  = [sol_rl["reward"][b * N].item() for b in range(B)]

        state = ([
            {"pred":  pred_per_sentence[b],
            "gold":  gold_per_sentence[b],
            "comp_reward": comp_reward_sent[b],
            "sol_reward":  sol_reward_sent[b]}
            for b in range(B)
        ] if is_test else {})

        return batch_forward, pred_per_sentence, gold_per_sentence, state



    def get_reward(self, pred_edges, gold_edges):
        """ Calculate the reward for the generated logical form by comparing to the gold."""
        len_gold = len(gold_edges)

        pred_unlabelled_set = set(['|'.join(edge[:2]) for edge in pred_edges])
        gold_unlabelled_set = set(['|'.join(edge[:2]) for edge in gold_edges])

        # composer reward is the fraction of correct pred edges out of the total number of gold edges
        composer_reward = len(pred_unlabelled_set & gold_unlabelled_set) / (len_gold)
        
        # solver reward is the fraction of correct pred labels out of the matching edges 
        # num_correct_labels / len(pred_unlabelled_set & gold_unlabelled_set)
        good_edges = [[pred_edge, gold_edge] for pred_edge in pred_edges for gold_edge in gold_edges if pred_edge[:2] == gold_edge[:2]]
        good_labels = sum(1 for pred_edge, gold_edge in good_edges if pred_edge[2] == gold_edge[2])
        
        solver_reward = good_labels / len(good_edges) if len(good_edges) > 0 else 0
        
        return composer_reward, solver_reward

    def action_info_dict(self, action_info):
        action_info_dict = {
            "embed_type": action_info[0],
            "agent": action_info[1],
            "theme": action_info[2],
            "recipient": action_info[3],
            "action_word": action_info[4]
        }
        return action_info_dict

    # Translate a nest class into a list
    def translate(self, head_sem):
        """ Takes a semantic class P and returns a flat list of edges.
        
        This is the final output of the model, and is used to generate the final logical form.
        """
        # we start with an empty chain of edges with shape [head, dep, rel_name]
        edges = []

        for child in head_sem.children:
            edges.append([head_sem.token, child.token, child.rel])
            # then we recursively call translate on the child
            edges += self.translate(child)
        return edges

    def process_gold(self, s):
        # Examples:
        # * cake ( x _ 8 ) ; cat ( x _ 1 ) AND cat . nmod ( x _ 1 , x _ 3 ) AND admire . agent ( x _ 3 , x _ 1 ) AND admire . ccomp ( x _ 3 , x _ 6 ) AND eat . agent ( x _ 6 , Emily ) AND eat . theme ( x _ 6 , x _ 8 )
        # * muffin ( x _ 4 ) ; * painting ( x _ 7 ) ; * girl ( x _ 10 ) ; mail . recipient ( x _ 2 , Emma ) AND mail . theme ( x _ 2 , x _ 4 ) AND mail . agent ( x _ 2 , x _ 10 ) AND muffin . nmod . beside ( x _ 4 , x _ 7 )

        # each edge will be a triplet of form (hed, dep, rel_name)
        # a predicate -> entity relation will look like (8, 1, "agent") (agent, theme, recipient)
        # a predicate -> predicate relation will look like (8, 6, "ccomp") (ccomp, xcomp)
        # an entity -> predicate relation will look like (1, 3, "relcl")
        # an entity -> entity relation will look like (1, 3, "in") (in, on, beside)
        edges = []
        variable_map = {}
        
        elements = s.replace(" and ", ";").split(";")
        for element in elements:
            # if there is an nmod, it can either be an nmod (if there is a preposition) or a relcl (if there is no preposition)
            # check for nmod first
            if ". nmod ." in element:
                # cake.nmod.on.x_4.x_7
                nmod = element.replace(" ", "").replace("(",".").replace(")","").replace(",",".")
                nmod = nmod.split(".")
                variable_map[nmod[3]] = nmod[0]
                nmod = nmod[-3:]
                # [on, x_4, x_7]
                nmod = [nmod[1], nmod[2], nmod[0]]
                nmod = [el.replace("x_", "") for el in nmod]
                # [4, 7, on]
                edges.append(nmod)

            # next, relcl
            elif ". nmod" in element:
                # cake.nmod.x_4.x_7
                relcl = element.replace(" ", "").replace("(",".").replace(")","").replace(",",".")
                relcl = relcl.split(".")
                variable_map[relcl[2]] = relcl[0]
                relcl = relcl[-2:]
                # [x_4, x_7]
                relcl = [el.replace("x_", "") for el in relcl] + ["relcl"]
                # [4, 7, relcl]
                edges.append(relcl)

            # else, if there is a comma, it is a predicate
            elif ',' in element:
                # study . agent ( x _ 2 , x _ 1 )
                predicate = element.replace(" ", "").replace("(",".").replace(")","").replace(",",".")
                # study.agent.x_2.x_1
                predicate = predicate.split(".")
                variable_map[predicate[2]] = predicate[0]
                predicate = predicate[-3:]
                # [agent, x_2, x_1]
                predicate = [predicate[1], predicate[2], predicate[0]]
                # [x_2, x_1, agent]
                predicate = [el.replace("x_", "") for el in predicate]
                # [2, 1, agent]
                edges.append(predicate)
                
            # if there is no comma, it is an entity, which we capture in variable_map
            else:
                # cake ( x _ 8 ) or * cake ( x _ 8 )
                element = element.replace(" ", "").replace("(",".").replace(")","").replace("*", "")
                # cake.x_8
                element = element.split(".")
                variable_map[element[1]] = element[0]

        # Now we go through the edges and replace the entities with their corresponding variables
        for i in range(len(edges)):
            for j in range(2):
                if edges[i][j].isnumeric():
                    assert "x_" + edges[i][j] in variable_map, f"Entity {edges[i][j]} not found in variable_map: {variable_map}"
                    edges[i][j] = variable_map["x_" + edges[i][j]]

        debug_info = {
            "elements": elements,
            "variable_map": variable_map
        }
        return edges, debug_info