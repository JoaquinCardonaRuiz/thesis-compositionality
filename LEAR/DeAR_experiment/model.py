import pdb
import random
import statistics
from itertools import chain

import math
import torch.nn.functional as F
from torch import nn
from masked_cross_entropy import *
from utils import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import clamp_grad
import time
import copy
import re
import pdb
import os

from modules.MSTDecoder import mst_decode

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

class Node:
    def __repr__(self,depth=0):
        s = f"{'\t'*depth}{self.rel} {self.__class__.__name__}({self.token}) - {self.ix}"
        for child in self.children:
            s += f"\n{child.__repr__(depth=depth+1)}"
        return s

    def __str__(self,depth=0):
        return self.__repr__(depth=depth)

class E(Node):  # Entities (M0~M9)
    # 0: in
    # 1: on
    # 2: beside
    # 3: relcl
    def __init__(self, entity, ix=-1):
        self.children = []
        self.token = entity
        self.rel = None
        self.ix = ix

    def add_child(self, child, rel):
        assert rel in ['in', 'on', 'beside', 'relcl']
        if rel == 'relcl':
            assert isinstance(child, P)
        else:
            assert isinstance(child, E)

        child_aux = copy.deepcopy(child)
        child_aux.rel = rel
        self.children.append(child_aux)




class P(Node):
    def __init__(self, action, ix=-1):
        self.children = []
        self.token = action
        self.rel = None
        self.ix = ix

        self.has_role = {"agent": False, "theme": False, "recipient": False}

        self.is_causative = False
        self.is_unaccusative = False

        self.is_full = False


    def add_child(self, child, rel):
        assert rel in ['ccomp', 'xcomp', 'agent', 'theme', 'recipient']
        if rel == 'ccomp' or rel == 'xcomp':
            assert isinstance(child, P)
        else:
            assert isinstance(child, E)
            if rel == 'agent':
                self.has_role['agent'] = True
            elif rel == 'theme':
                self.has_role['theme'] = True
            elif rel == 'recipient':
                self.has_role['recipient'] = True
            if self.has_role['agent'] and \
                    self.has_role['theme'] and \
                    self.has_role['recipient']:
                self.is_full = True

        child_aux = copy.deepcopy(child)
        child_aux.rel = rel
        self.children.append(child_aux)


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


class DepTreeComposer(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, input_lang,
                 output_lang, alignments_idx={}, entity_list=[],
                 caus_predicate_list=[], unac_predicate_list=[],
                 dropout_prob=None, context_type="bilstm", num_layers=2, num_heads=4):

        super().__init__()

        self.embd_parser = nn.Embedding(vocab_size, input_dim)
        self.hidden_dim = hidden_dim
        self.context_type = context_type.lower()
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.alignments_idx = alignments_idx
        self.entity_list = entity_list
        self.predicate_list = caus_predicate_list + unac_predicate_list

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else nn.Identity()

        self.arc_mlp_head = nn.Linear(hidden_dim, hidden_dim)
        self.arc_mlp_dep = nn.Linear(hidden_dim, hidden_dim)

        if self.context_type == "bilstm":
            self.context_encoder = nn.LSTM(hidden_dim, hidden_dim // 2,
                                           num_layers=1, batch_first=True, bidirectional=True)
        elif self.context_type == "transformer":
            encoder_layer = TransformerEncoderLayer(d_model=hidden_dim,
                                                    nhead=num_heads,
                                                    dim_feedforward=hidden_dim * 4,
                                                    dropout=dropout_prob or 0.1,
                                                    batch_first=True)
            self.context_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown context_type: {context_type}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.arc_mlp_head.weight)
        nn.init.xavier_uniform_(self.arc_mlp_dep.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def encode_contextual(self, token_embeddings):
        x = self.linear(token_embeddings.unsqueeze(0))  # (1, S, D)
        x = self.dropout(x)

        if self.context_type == "bilstm":
            output, _ = self.context_encoder(x)
        elif self.context_type == "transformer":
            output = self.context_encoder(x)
        return output.squeeze(0)  # (S, D)

    def score_arcs(self, hidden):
        head_repr = self.arc_mlp_head(hidden)  # (S, D)
        dep_repr = self.arc_mlp_dep(hidden)    # (S, D)
        arc_scores = torch.matmul(head_repr, dep_repr.T)  # (S, S)
        return arc_scores

    def forward(self, x, bottom_idx):
        """
        Args:
            x: (seq_len,) Tensor of token indices

        Returns:
            normalized_entropy: dummy tensor for API consistency
            log_prob: dummy tensor for API consistency
            head_dep_idx: list of (head, dependent) tuples
        """
        token_embeds = self.embd_parser(x)  # (S, input_dim)

        contextualized = self.encode_contextual(token_embeds)  # (S, hidden_dim)
        
        # bottom_idx is a list containing only the indices of the important tokens
        # we only keep the embeddings of the important tokens
        bottom_contextualized = contextualized[bottom_idx]  # (B, hidden_dim) 

        # TODO: Replace with recurrent approach
        arc_scores = self.score_arcs(bottom_contextualized)  # (B, B)

        heads = mst_decode(arc_scores)  # List[int]
        head_dep_idx = [(heads[i], i) for i in range(len(heads))]

        # map to original indices
        head_dep_idx = [(bottom_idx[head], bottom_idx[dep]) for head, dep in head_dep_idx]

        idx2contextual = {str(bottom_idx[i]): bottom_contextualized[i] for i in range(bottom_contextualized.size(0))}
        
        debug_info = {
            "x_input": x,
            "bottom_idx": bottom_idx,
            "arc_scores": arc_scores,
            "heads": heads,
        }
        return head_dep_idx, idx2contextual, debug_info


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
        self.semantic_E_E = nn.Linear(in_features=hidden_dim, out_features=3)

        # 0: agent
        # 1: theme
        # 2: recipient
        self.semantic_P_E = nn.Linear(in_features=hidden_dim, out_features=3)

        # 0: ccomp
        # 1: xcomp
        self.semantic_P_P = nn.Linear(in_features=hidden_dim, out_features=2)

        
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

    def forward(self, pair, head_dep_idx, idx2contextual, idx2output_token):
        # returns either E or P instances depending of whether the token is on the entity or predicate list
        # idx2output_token is a list of pairs like [ 1, "cake"], [3, "cook"]]
        # idx2semantic is a dictionary of pairs like {"1": E("cake"), "3": P("cook")}
        idx2semantic = self.init_semantic_class(idx2output_token)
        # pdb.set_trace()

        semantic_normalized_entropy = []
        semantic_log_prob = []

        # Example tree: [(1, 1), (8, 2), (8, 3), (8, 5), (1, 8)]
        # Example idx2semantic: {"1": E("cake"), "3": P("cook")}

        # Sort the head_dep_idx in a bottom-up manner (and remove root self-pair)
        root_idx = [head for head, dep in head_dep_idx if head == dep][0]
        head_dep_idx = self.sort_bottom_up(head_dep_idx)

        for head, dep in head_dep_idx:
            # get semantic class of the head and dependent
            head_sem = idx2semantic[str(head)]
            dep_sem = idx2semantic[str(dep)]

            head_contextual = idx2contextual[str(head)]

            cat_distr, _, actions_i, parent_semantic = self.semantic_merge(head_sem, dep_sem, head_contextual) 

            idx2semantic[str(head)] = parent_semantic

            if cat_distr is not None:
                # collect for RL
                semantic_normalized_entropy.append(cat_distr.normalized_entropy)
                semantic_log_prob.append(-cat_distr.log_prob(actions_i))

        # calculate the normalized entropy and log probability of the actions taken for RL
        assert len(semantic_normalized_entropy) > 0, f"===== No semantic actions taken =====\n{pair}\n{semantic_normalized_entropy}\n{idx2semantic}"
        normalized_entropy = sum(semantic_normalized_entropy) / len(semantic_normalized_entropy)
        semantic_log_prob = sum(semantic_log_prob)

        # the final meaning is in the root (the last head idx)
        assert head == root_idx, "Error in the tree structure, the root is not the last head idx"
        final_semantic = copy.deepcopy(idx2semantic[str(head)]) 

        # if final semantic is an entity, we create a ghost predicate for it
        if isinstance(final_semantic, E):
            ghost_pred = P('ghost_predicate')
            ghost_pred.children = [final_semantic]
            final_semantic = copy.deepcopy(ghost_pred)
        
        # TODO: deal with this
        '''
        # fix the representation of unnaccusative verbs (such as fall), where the subject is actually the theme
        for action_idx in range(len(final_semantic.action_chain)):
            action_info = final_semantic.action_chain[action_idx]
            if action_info[1] is not None and action_info[2] is None and action_info[3] is None:
                agent_class = copy.deepcopy(action_info[1])
                assert agent_class.entity_chain[0][1] == 'agent'
                action = action_info[4]

                # in gen
                # A donut on the bed rolled .
                # * bed ( x _ 4 ) ; donut ( x _ 1 ) AND donut . nmod . on ( x _ 1 , x _ 4 ) AND roll . theme ( x _ 5 , x _ 1 )
                # The girl beside a table rolled .
                # * girl ( x _ 1 ) ; girl . nmod . beside ( x _ 1 , x _ 4 ) AND table ( x _ 4 ) AND roll . agent ( x _ 5 , x _ 1 )

                # in train
                # Victoria hoped that a girl rolled .
                # hope . agent ( x _ 1 , Victoria ) AND hope . ccomp ( x _ 1 , x _ 5 ) AND girl ( x _ 4 ) AND roll . theme ( x _ 5 , x _ 4 )

                if action in self.unac_predicate_list and len(agent_class.entity_chain) == 1:
                    # if action in self.unac_predicate_list:
                    agent_class.entity_chain[0][1] = 'theme'
                    final_semantic.action_chain[action_idx][1] = None
                    final_semantic.action_chain[action_idx][2] = agent_class
                    final_semantic.has_agent = False
                    final_semantic.has_theme = True
        # update the span2semantic with the fixed final semantic
        idx2semantic[str(parent_span)] = final_semantic
        '''
        debug_info = {"idx2semantic": idx2semantic,
                      "head_dep_idx": head_dep_idx,
                      "root_idx": root_idx,
                      "last_head": head}

        return final_semantic, normalized_entropy, semantic_log_prob, debug_info

    def init_semantic_class(self, idx2output_token):
        idx2semantic = {}
        for idx in idx2output_token:
            idx_position = idx[0]
            output_token = idx[1]

            assert output_token in self.entity_list + self.predicate_list

            if output_token in self.entity_list:
                idx_semantic = E(output_token, idx_position)
            else:
                idx_semantic = P(output_token, idx_position)

            idx2semantic[str(idx_position)] = idx_semantic

        return idx2semantic

    def semantic_merge(self, head_sem, dep_sem, head_contextual):
        """ Compose the two children into a parent semantic class. 
        
        This is done by selecting a semantic operation from the semantic classifier.
        """
        # E2E classifier
        if isinstance(head_sem, E) and isinstance(dep_sem, E):
            semantic_score = self.semantic_E_E(head_contextual)
            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, #
                                                        semantic_mask, 
                                                        self.relaxed, 
                                                        self.tau_weights,
                                                        self.straight_through,
                                                        self.gumbel_noise)
            # 0: in, 1: on, 2: beside
            parent_semantic = copy.deepcopy(head_sem)
            if actions[0] == 1:
                parent_semantic.add_child(dep_sem, 'in')
            elif actions[1] == 1:
                parent_semantic.add_child(dep_sem, 'on')
            elif actions[2] == 1:
                parent_semantic.add_child(dep_sem, 'beside')
            else:
                raise ValueError(f"Invalid action by E2E classifier: {actions}")

        # P2P classifier
        elif isinstance(head_sem, P) and isinstance(dep_sem, P):
            semantic_score = self.semantic_P_P(head_contextual)
            semantic_mask = torch.ones_like(semantic_score)

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, #
                                                        semantic_mask, 
                                                        self.relaxed, 
                                                        self.tau_weights,
                                                        self.straight_through,
                                                        self.gumbel_noise)
            # 0: ccomp, 1: xcomp
            parent_semantic = copy.deepcopy(head_sem)
            if actions[0] == 1:
                parent_semantic.add_child(dep_sem, 'ccomp')
            elif actions[1] == 1:
                assert actions[1] == 1
                parent_semantic.add_child(dep_sem, 'xcomp')
            else:
                raise ValueError(f"Invalid action by P2P classifier: {actions}")

        # P2E classifier
        elif isinstance(head_sem, P) and isinstance(dep_sem, E):
            semantic_score = self.semantic_P_E(head_contextual)
            semantic_mask = torch.ones_like(semantic_score)

            if not parent_semantic.is_full:
                assert parent_semantic.has_agent is False or \
                        parent_semantic.has_theme is False or \
                        parent_semantic.has_recipient is False
                if parent_semantic.has_agent is True:
                    semantic_mask[0] = 0
                if parent_semantic.has_theme is True:
                    semantic_mask[1] = 0
                if parent_semantic.has_recipient is True:
                    semantic_mask[2] = 0
            else:
                # TODO: what to do when the parent is full?
                pass

            cat_distr = Categorical(semantic_score, semantic_mask)
            actions, gumbel_noise = self._sample_action(cat_distr, #
                                                    semantic_mask, 
                                                    self.relaxed, 
                                                    self.tau_weights,
                                                    self.straight_through,
                                                    self.gumbel_noise)
            # 0: agent, 1: theme, 2: recipient
            parent_semantic = copy.deepcopy(head_sem)
            if actions[0] == 1:
                parent_semantic.add_child(dep_sem, 'agent')
            elif actions[1] == 1:
                parent_semantic.add_child(dep_sem, 'theme')
            elif actions[2] == 1:
                assert actions[2] == 1
                parent_semantic.add_child(dep_sem, 'recipient')
            else:
                raise ValueError(f"Invalid action by P2E classifier: {actions}")
            
        # E2P classifier
        elif isinstance(head_sem, E) and isinstance(dep_sem, P):
            # No classifier for this case
            # we know its a relcl
            parent_semantic = copy.deepcopy(head_sem)
            parent_semantic.add_child(dep_sem, 'relcl')
            cat_distr, gumbel_noise, actions = None, None, None

        else:
            raise ValueError(f"Invalid semantic class: {head_sem}, {dep_sem}")

        return cat_distr, gumbel_noise, actions, parent_semantic


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
        self.abstractor = BottomAbstrator(alignments_idx)
        self.classifier = BottomClassifier(output_lang, alignments_idx)
        self.composer = DepTreeComposer(
            input_dim=word_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            input_lang=input_lang,
            output_lang=output_lang,
            alignments_idx=alignments_idx,
            entity_list=entity_list,
            caus_predicate_list=caus_predicate_list,
            unac_predicate_list=unac_predicate_list,
            dropout_prob=None,
            context_type="bilstm",
        )
        self.solver = Solver(hidden_dim, output_lang,
                             entity_list=entity_list,
                             caus_predicate_list=caus_predicate_list,
                             unac_predicate_list=unac_predicate_list)
        self.var_norm_params = {"var_normalization": var_normalization, "var": 1.0, "alpha": 0.9}
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.reset_parameters()
        self.is_test = False

        self.case = None

    # TODO: change the paremeters
    def get_high_parameters(self):
        return list(chain(self.composer.parameters()))

    def get_low_parameters(self):
        return list(chain(self.solver.parameters()))

    def forward(self, pair, x, sample_num, is_test=False, epoch=None):
        self.is_test = is_test
        batch_forward_info, pred_chain, label_chain, state = self._forward(
            pair, x, sample_num, epoch, is_test)
        return batch_forward_info, pred_chain, label_chain, state

    def _forward(self, pair, x, sample_num, epoch, is_test):
        debug_info = {}

        assert x.size(0) > 1
        # [A] [cake] [was] [cooked] [by] [the] [scientist]

        bottom_idx = self.abstractor(x)
        # [1, 3, 6]
        # Corresponding to: cake, cooked, scientist

        idx2output_token = self.classifier(x, bottom_idx)
        # [[1, "cake"], [3, "cook"], [6, "scientist"]]

        # repeat sample_num (10) times to explore different trees
        #bottom_idx_batch = [bottom_idx for _ in range(sample_num)]
        #idx2output_token_batch = [idx2output_token for _ in range(sample_num)]

        # tree_rl_info has one entry per sample of the batch
        # it contains the normalized entropy, log prob, parent-child pairs (the merges that were made) and idx2repre (embeddings of tokens)
        # span2variable_batch contains a mapping of the abstracted spans ([1, 1]) to the tokens (cake, cooked, scientist)
        head_dep_idx, idx2contextual, debug_info["composer"] = self.composer(x, bottom_idx)

        batch_forward_info = []

        # we iterate over samples of the batch, and apply semantic composition
        for in_batch_idx in range(sample_num):
            # the solvers walks the tree bottom up, applies semantic composition rules, and generates a logical form
            final_semantic, normalized_entropy, semantic_log_prob, sem_debug_info = self.solver(pair, 
                                                                                                head_dep_idx, 
                                                                                                idx2contextual, 
                                                                                                idx2output_token)

            # debug prints
            print(f"===== Input Sentence =====\n{pair[0]}")
            print(f"===== Gold Label =====\n{pair[1]}")
            print(f"===== Input Tree =====\n{head_dep_idx}")
            print(f"===== Nodes =====\n{bottom_idx}")
            print(f"===== Alignment =====\n{idx2output_token}")
            print(f"===== Final Semantic =====\n{final_semantic}")
            # end debug

            # conver the gold label into a flat chain of tokens representing the logical form
            label_chain, gold_debug_info = self.process_output(pair[1])
            print(f"===== Gold Label Chain =====\n{label_chain}")
            quit()

            # convert the resulting tree structure into a flat chain of tokens representing the logical form
            pred_chain = self.translate(span2semantic[str(end_span)])
            # pdb.set_trace()
            

            #if "who" in pair[1] or "what" in pair[1]:
            #    print("")
            #    print(f"Gold Label: {pair[1]}")
            #    print(f"Parsed Gold Label {label_chain}")
            #    print(f"Predicted Label Chain {pred_chain}")
            #Gold Label: * table ( x _ 6 ) ; cook . agent ( x _ 1 , ? ) and cook . theme ( x _ 1 , x _ 3 ) and cake ( x _ 3 ) and cake . nmod . beside ( x _ 3 , x _ 6 )
            #Parsed Gold Label ['cook', '?', 'cake', 'beside', 'table', 'None']
            #Predicted Label Chain ['cook', 'who', 'cake', 'beside', 'table', 'None']

            # and calculate the reward for the generated logical form by comparing to the gold
            reward = self.get_reward(pred_chain, label_chain)

            # pdb.set_trace()

            # combine the syntactic (from the composer) and semantic (from the solver) information to train jointly
            normalized_entropy = tree_normalized_entropy[in_batch_idx] + \
                                 semantic_normalized_entropy

            log_prob = tree_log_prob[in_batch_idx] + \
                       semantic_log_prob

            batch_forward_info.append([normalized_entropy, log_prob, reward])

            '''
            if pair[0] == 'a visitor gave a cookie to the girl that emma rolled':
                import pickle
                with open("debug_info.pkl", "wb") as f:
                    pickle.dump(span2semantic[str(end_span)], f)
                print("Saved debug info")
                print(pred_chain)
                quit()
            '''            
        # pdb.set_trace()
        state = {
            "bottom_span": bottom_span,
            "span2output_token": span2output_token_batch,
            "parent_child_spans": parent_child_spans,
            "span2output_token": span2output_token,
            "span2semantic": span2semantic,
            "end_span": end_span,
            "pred_chain": pred_chain,
            "label_chain": label_chain,
            "pair": pair,
            "parent_json": span2semantic[str(end_span)].json_repr
        }

        return batch_forward_info, pred_chain, label_chain, state


    def get_reward(self, pred_chain, label_chain):
        """ Calculate the reward for the generated logical form by comparing to the gold."""
        # get lens
        pred_len = len(pred_chain)
        label_len = len(label_chain)

        # longest common consecutive subsequence
        max_com_len = 0

        # for every token in the model's output
        for pred_idx in range(pred_len):
            # for every token in the gold label
            for label_idx in range(label_len):
                com_len = 0
                if pred_chain[pred_idx] != label_chain[label_idx]:
                    continue
                else:
                    com_len += 1
                    right_hand_length = min(pred_len - pred_idx - 1, label_len - label_idx - 1)
                    for right_hand_idx in range(right_hand_length):
                        if pred_chain[pred_idx + right_hand_idx + 1] == label_chain[label_idx + right_hand_idx + 1]:
                            com_len += 1
                            continue
                        else:
                            break
                    if com_len > max_com_len:
                        max_com_len = com_len

        reward = max_com_len / (pred_len + label_len - max_com_len)

        assert reward <= 1.
        if reward == 1.:
            assert pred_chain == label_chain

        return reward

    def get_entity_chain(self, semantic):
        if semantic is None:
            return ['None']

        assert isinstance(semantic, E)
        flat_chain = []
        # [in_word / on_word / beside_word, agent_word / theme_word / recipient_word, entity]
        for entity_idx, entity_info in enumerate(semantic.entity_chain):
            if entity_idx == 0:
                flat_entity_chain = [entity_info[2]]
            else:
                assert entity_info[0] is not None
                flat_entity_chain = [entity_info[0], entity_info[2]]
            flat_chain = flat_chain + flat_entity_chain

        return flat_chain

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
    def translate(self, semantic):
        """ Takes a semantic class P and returns a flat list of tokens.
        
        This is the final output of the model, and is used to generate the final logical form.
        """
        # ensure tree root is a predicate
        assert isinstance(semantic, P)

        # we start with an empty chain of tokens, and iterate over the action chain of the tree
        flat_chain = []

        for action_idx, action_info in enumerate(semantic.action_chain):
            # action_info is a list of the form:
            # [action_type, agent_obj, theme_obj, recipient_obj, action_word]

            # [ccomp_word / xcomp_word, agent_entity_class, theme_entity_class, recipient_entity_class, action_word]
            
            # our first action is always a predicate, so we start with the action word
            if action_idx == 0:
                flat_action_chain = [action_info[4]]
            # later actions have either a ccomp or xcopm, and then the action word
            else:
                assert action_info[0] is not None
                flat_action_chain = [action_info[0], action_info[4]]

            # if it is an xcomp, we use the previous action word as the agent for this action
            # xcomp is for structures such as "the girl tried to roll", where the subjects of
            # both "tried" and "roll" are the same
            if action_info[0] == 'xcomp':
                agent_chain = reserve_agent_chain
            else:
                agent_chain = self.get_entity_chain(action_info[1])
            # we use get_entity_chain to flatten entities into its string components
            theme_chain = self.get_entity_chain(action_info[2])
            recipient_chain = self.get_entity_chain(action_info[3])

            # compose a flat action chain from the action word and the agent, theme, and recipient chains
            flat_action_chain = flat_action_chain + agent_chain + theme_chain + recipient_chain
            reserve_agent_chain = copy.deepcopy(agent_chain)

            # append the flat action chain to the flat chain
            flat_chain = flat_chain + flat_action_chain

        return flat_chain

    def process_output(self, s):
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