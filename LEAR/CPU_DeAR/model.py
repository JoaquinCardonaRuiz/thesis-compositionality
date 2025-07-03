from itertools import chain

from torch import nn
from masked_cross_entropy import *
from utils import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import clamp_grad
import copy
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
        self.rel = ''
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
        self.rel = ''
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



    def forward(self, x_embedding, bottom_idx):
        """ Forward pass of the DepTreeComposer.

        Builds a dependency tree from the input sequence x, which is a sequence of token embeddings.
        """
        # debug prints
        #print(f"Bottom indices: {bottom_idx}")

        # Define edges and rl info
        log_probs = []
        entropies = []
        edges = []

        B = len(bottom_idx)
        #print(f"Batch size: {B}")
        mask = torch.ones((x_embedding.size(0),), dtype=torch.float32, device=x_embedding.device)
        #print(f"Mask: {mask}")

        
        if USE_CUDA:
            mask = mask.cuda()

        # next, we get the leaf embeddings of the input sequence (all tokens)
        # this will be a hidden state and a cell state for each token in the input sequence
        hidden_1, cell_1 = self._transform_leafs(x_embedding, mask)
        hidden = hidden_1[bottom_idx]  # (B, D)
        cell = cell_1[bottom_idx]  # (B, D)
        mask = mask[bottom_idx]

        # Initial contextual embeddings
        idx2contextual = {
            str(bottom_idx[i]): hidden[i] for i in range(B)
        }
        
        # debug prints
        #print(f"Initial hidden state: {hidden.size()}, cell state: {cell.size()}")

        edges = []
        while len(edges) < len(bottom_idx) - 1: 

            cat_distr, _, actions, hidden, cell = self._make_step(hidden, cell, mask)
            
            actions_idx_tensor = actions.argmax()
            actions_idx = actions_idx_tensor.item()
            B = hidden.size(0)
            head_idx, dep_idx = divmod(actions_idx, B)

            # Update contextual representation for the dependent
            idx2contextual[str(bottom_idx[head_idx])] = hidden[head_idx]

            log_probs.append(cat_distr.log_prob(actions_idx_tensor))
            entropies.append(cat_distr.entropy)

            edges.append((bottom_idx[head_idx], bottom_idx[dep_idx]))
            mask[dep_idx] = 0  # Mark dep as attached
            # debug prints
            #print(f"Current edges: {edges}")
            #print(f"Action taken: {actions_idx} (head: {head_idx}, dep: {dep_idx})")
            #print(f"Updated mask: {mask}")

        # Package outputs
        rl_info = {
            "normalized_entropy": torch.stack(entropies).mean().unsqueeze(0),
            "log_prob": torch.stack(log_probs).sum().unsqueeze(0)
        }

        # debug prints
        #print(f"Edges: {edges}")
        #print(f"RL Info: {rl_info}")

        return edges, idx2contextual, rl_info
    
    def _make_step(self, hidden, cell, mask_ori):
        """
        One REINFORCE step: score all (head, dep) pairs and update selected head.
        
        Args:
            hidden: (B, D)
            cell: (B, D)
            mask_ori: (B,) float — 1 if dep is unassigned

        Returns:
            cat_distr: Categorical distribution over valid arcs
            gumbel_noise: Updated noise (if used)
            actions: One-hot vector over (B*B)
            hidden: (B, D) updated hidden states
            cell: (B, D) updated cell states
        """
        B, D = hidden.size()
        device = hidden.device
        mask = copy.deepcopy(mask_ori)

        # Build head × dep matrices
        head_h = hidden.unsqueeze(1).expand(B, B, D)  # (B, B, D)
        head_c = cell.unsqueeze(1).expand(B, B, D)
        dep_h = hidden.unsqueeze(0).expand(B, B, D)
        dep_c = cell.unsqueeze(0).expand(B, B, D)

        # Compose new head representations based on head←dep
        new_h, new_c = self.tree_lstm_cell(
            head_h.reshape(-1, D),
            head_c.reshape(-1, D),
            dep_h.reshape(-1, D),
            dep_c.reshape(-1, D)
        )
        new_h = new_h.view(B, B, D)
        new_c = new_c.view(B, B, D)

        # Score all head→dep arcs
        arc_features = torch.cat([head_h, dep_h], dim=-1)  # (B, B, 2D)
        score = self.arc_scorer(arc_features).squeeze(-1)  # (B, B)

        # Mask invalid arcs: prevent attaching to already attached deps
        arc_mask = torch.ones_like(score, dtype=torch.float32, device=device)
        arc_mask.fill_diagonal_(0)  # no self-loops
        arc_mask *= mask.unsqueeze(0)  # block already-attached deps (on dep axis)

        # Flatten scores and masks for categorical sampling
        flat_score = score.view(-1)
        flat_mask = arc_mask.view(-1)

        # Sample from categorical distribution over valid arcs
        cat_distr = Categorical(flat_score, flat_mask)
        actions, gumbel_noise = self._sample_action(cat_distr, flat_mask)
        actions_2d = actions.view(B, B)  # (B, B)

        # Select updated representations for heads
        new_h_updated = (actions_2d.unsqueeze(-1) * new_h).sum(dim=1)  # (B, D)
        new_c_updated = (actions_2d.unsqueeze(-1) * new_c).sum(dim=1)  # (B, D)

        # Update only the selected heads — leave others unchanged
        update_mask = actions_2d.sum(dim=1, keepdim=True)  # shape (B, 1), 1 for selected head
        hidden = hidden * (1 - update_mask) + new_h_updated * update_mask
        cell = cell * (1 - update_mask) + new_c_updated * update_mask

        return cat_distr, gumbel_noise, actions, hidden, cell


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

    def forward(self, pair, head_dep_idx, idx2contextual, idx2output_token):
        debug_info = {}

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

        # Find the root index: the only index that does not appear as a dependent
        #print(f"head_dep_idx: {head_dep_idx}")
        #root_idx = set(head for head, _ in head_dep_idx) - set(dep for _, dep in head_dep_idx)
        #print(f"Root index: {root_idx}")
        #assert len(root_idx) == 1, f"Expected exactly one root, got {root_idx} from {head_dep_idx}"
        #root_idx = list(root_idx)[0]
        #debug_info["root_idx"] = root_idx

        #head_dep_idx = self.sort_bottom_up(head_dep_idx)
        #debug_info["sorted_head_dep_idx"] = head_dep_idx

        for head, dep in head_dep_idx:
            # get semantic class of the head and dependent
            #print(head_dep_idx)
            head_sem = idx2semantic[str(head)]
            dep_sem = idx2semantic[str(dep)]

            head_contextual = idx2contextual[str(head)]
            dep_contextual = idx2contextual[str(dep)]
            #print(f"head norm: {head_contextual.norm().item()}, dep norm: {dep_contextual.norm().item()}")
            semantic_input = torch.cat([head_contextual, dep_contextual], dim=-1)
            

            cat_distr, _, actions_i, chosen_rel = self.semantic_merge(head_sem, dep_sem, semantic_input) 

            semantic_edges.append((str(head), str(dep), chosen_rel))
            #print(f"Semantic edges so far: {semantic_edges}")

            if cat_distr is not None:
                # collect for RL
                semantic_normalized_entropy.append(cat_distr.normalized_entropy)
                semantic_log_prob.append( cat_distr.log_prob(actions_i))

        debug_info["idx2semantic"] = idx2semantic
        debug_info["semantic_edges"] = semantic_edges

        # calculate the normalized entropy and log probability of the actions taken for RL
        if len(semantic_normalized_entropy) == 0:
            # No stochastic decision was made in this sample
            device = idx2contextual[str(head)].device      # ← swap in for x.device
            normalized_entropy = torch.tensor(0.0, device=device)
            semantic_log_prob  = torch.tensor(0.0, device=device)
        else:
            normalized_entropy = sum(semantic_normalized_entropy) / len(semantic_normalized_entropy)
            semantic_log_prob  = sum(semantic_log_prob)

        # the final meaning is in the root (the last head idx)
        #assert head == root_idx, f"Error in the tree structure, the root is not the last head idx.\n\n\n {debug_info}"
        #final_semantic = copy.deepcopy(idx2semantic[str(head)]) 
        
        '''
        # if final semantic is an entity, we create a ghost predicate for it
        if isinstance(final_semantic, E):
            ghost_pred = P('ghost_predicate')
            ghost_pred.children = [final_semantic]
            final_semantic = copy.deepcopy(ghost_pred)
        '''
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
        rl_info = {
            "normalized_entropy": normalized_entropy.unsqueeze(0),
            "log_prob": semantic_log_prob.unsqueeze(0),
        }
                      

        return semantic_edges, rl_info, debug_info

    def init_semantic_class(self, idx2output_token):
        idx2semantic = {}
        for idx in idx2output_token:
            idx_position = idx[0]
            output_token = idx[1]

            assert output_token in self.entity_list + self.predicate_list

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
            #print(f"Softmax probs (E,E): {F.softmax(semantic_score, dim=-1)}")

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
            #print(f"Softmax probs (P,P): {F.softmax(semantic_score, dim=-1)}")
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
            #print(f"Softmax probs (P,E): {F.softmax(semantic_score, dim=-1)}")
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

    # TODO: change the paremeters
    def get_high_parameters(self):
        return list(chain(self.composer.parameters())) + list(chain(self.embd_parser.parameters()))

    def get_low_parameters(self):
        return list(chain(self.solver.parameters()))

    def forward(self, pair, x, sample_num, is_test=False, epoch=None):
        self.is_test = is_test
        batch_forward_info, pred_chain, label_chain, state = self._forward(
            pair, x, sample_num, epoch, is_test)
        return batch_forward_info, pred_chain, label_chain, state

    def _forward(self, pair, x, sample_num, epoch, is_test):
        debug_info = {}
        assert x.size(0) > 1, f"Input sequence length should be greater than 1. \n{x}\n{pair}\n{x.size()}"
        # [A] [cake] [was] [cooked] [by] [the] [scientist]
    
        bottom_idx = self.abstractor(x)
        # [1, 3, 6]
        # Corresponding to: cake, cooked, scientist

        idx2output_token = self.classifier(x, bottom_idx)
        # [[1, "cake"], [3, "cook"], [6, "scientist"]]

        # repeat sample_num (10) times to explore different trees
        #bottom_idx_batch = [bottom_idx for _ in range(sample_num)]
        #idx2output_token_batch = [idx2output_token for _ in range(sample_num)]

        batch_forward_info = []
        
        # Get embeddings
        #positions = torch.arange(len(x), device=x.device)
        #position_embed = self.position_embedding(positions)
        x_embedding = self.embd_parser(x)# + position_embed


        # we iterate over samples of the batch, and apply semantic composition
        for in_batch_idx in range(sample_num):
            #print(f"Pair: {pair}")
            # it contains the normalized entropy, log prob, parent-child pairs (the merges that were made) and idx2repre (embeddings of tokens)
            # span2variable_batch contains a mapping of the abstracted spans ([1, 1]) to the tokens (cake, cooked, scientist)
            '''
            rl_info = {
                "normalized_entropy": torch.stack(entropies).mean(),
                "log_prob": torch.stack(log_probs).sum()
            }
            '''
            edges, idx2contextual, composer_rl_info = self.composer(x_embedding, bottom_idx)


            # the solvers walks the tree bottom up, applies semantic composition rules, and generates a logical form
            #print(bottom_idx)
            semantic_edges, solver_rl_info, debug_info["solver"] = self.solver(pair, 
                                                                        edges, 
                                                                        idx2contextual, 
                                                                        idx2output_token)
            #print(semantic_edges)
            #total_log_prob = composer_rl_info["log_prob"] + solver_rl_info["log_prob"]
            #total_entropy = composer_rl_info["normalized_entropy"] + solver_rl_info["normalized_entropy"]

            # convert the resulting tree structure into a flat chain of tokens representing the logical form
            #pred_edges = self.translate(final_semantic)
            #print(f"idx2output_token: {idx2output_token}")
            #print(f"Semantic edges: {semantic_edges}")
            pred_edges = [[str(idx2output_token[[i[0] for i in idx2output_token].index(int(edge[0]))][1]), str(idx2output_token[[i[0] for i in idx2output_token].index(int(edge[1]))][1]), edge[2]] for edge in semantic_edges]
            #print(f"Predicted edges: {pred_edges}")
            # conver the gold label into a flat chain of tokens representing the logical form
            gold_edges, debug_info["process_gold"] = self.process_gold(pair[1])
            #print(f"Gold edges: {gold_edges}")

            #if "who" in pair[1] or "what" in pair[1]:
            #    print("")
            #    print(f"Gold Label: {pair[1]}")
            #    print(f"Parsed Gold Label {label_chain}")
            #    print(f"Predicted Label Chain {pred_chain}")
            #Gold Label: * table ( x _ 6 ) ; cook . agent ( x _ 1 , ? ) and cook . theme ( x _ 1 , x _ 3 ) and cake ( x _ 3 ) and cake . nmod . beside ( x _ 3 , x _ 6 )
            #Parsed Gold Label ['cook', '?', 'cake', 'beside', 'table', 'None']
            #Predicted Label Chain ['cook', 'who', 'cake', 'beside', 'table', 'None']

            # and calculate the reward for the generated logical form by comparing to the gold
            composer_rl_info['reward'], solver_rl_info['reward'] = self.get_reward(pred_edges, gold_edges)
            #print(f"Composer reward: {composer_rl_info['reward']}, Solver reward: {solver_rl_info['reward']}")
            #print(f"Reward: {reward}")
            batch_forward_info.append([composer_rl_info, solver_rl_info])

            '''
            if pair[0] == 'a visitor gave a cookie to the girl that emma rolled':
                import pickle
                with open("debug_info.pkl", "wb") as f:
                    pickle.dump(span2semantic[str(end_span)], f)
                print("Saved debug info")
                print(pred_chain)
                quit()
            '''   

            #print(debug_info["composer"])
            #print(debug_info["solver"])
            #print(f"Predicted edges: {pred_edges}")
            #print(f"Gold edges: {gold_edges}")
            #print(f"Reward: {reward}")         
            #quit()
        # pdb.set_trace()
        state = {
            "bottom_idx": bottom_idx,
            #"comp_heads": debug_info["composer"]["heads"],
            #"head_dep_edges": debug_info["solver"]["sorted_head_dep_idx"],
            "pred_edges": pred_edges,
            "gold_edges": gold_edges,
            "pair": pair,
            "comp reward": composer_rl_info['reward'],
            "solv reward": solver_rl_info['reward']
        }
        #print(f"==========\n{state}\n")
        #quit()
        return batch_forward_info, pred_edges, gold_edges, state


    def get_reward(self, pred_edges, gold_edges):
        """ Calculate the reward for the generated logical form by comparing to the gold."""
        len_gold = len(gold_edges)
        pred_set = set(['|'.join(edge) for edge in pred_edges])
        gold_set = set(['|'.join(edge) for edge in gold_edges])

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