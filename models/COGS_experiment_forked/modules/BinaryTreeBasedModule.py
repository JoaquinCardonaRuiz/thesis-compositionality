import torch
from torch import nn

from modules.BinaryTreeLstmCell import BinaryTreeLstmCell
from modules.LstmRnn import LstmRnn


class BinaryTreeBasedModule(nn.Module):
    no_transformation = "no_transformation"
    lstm_transformation = "lstm_transformation"
    bi_lstm_transformation = "bi_lstm_transformation"
    conv_transformation = "conv_transformation"

    def __init__(self, input_dim, hidden_dim, leaf_transformation, trans_hidden_dim, dropout_prob):
        super().__init__()
        self.leaf_transformation = leaf_transformation
        if leaf_transformation == BinaryTreeBasedModule.no_transformation:
            self.linear = nn.Linear(in_features=input_dim, out_features=2 * hidden_dim)
        elif leaf_transformation == BinaryTreeBasedModule.lstm_transformation:
            self.lstm = LstmRnn(input_dim, trans_hidden_dim)
            self.linear = nn.Linear(in_features=trans_hidden_dim, out_features=2 * hidden_dim)
        elif leaf_transformation == BinaryTreeBasedModule.bi_lstm_transformation:
            self.lstm_f = LstmRnn(input_dim, trans_hidden_dim)
            self.lstm_b = LstmRnn(input_dim, trans_hidden_dim)
            self.linear = nn.Linear(in_features=2 * trans_hidden_dim, out_features=2 * hidden_dim)
        elif leaf_transformation == BinaryTreeBasedModule.conv_transformation:
            self.conv1 = nn.Conv1d(input_dim, trans_hidden_dim, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(trans_hidden_dim, trans_hidden_dim, kernel_size=3, padding=1)
            self.linear = nn.Linear(in_features=trans_hidden_dim, out_features=2 * hidden_dim)
        else:
            raise ValueError(f'"{leaf_transformation}" is not in the list of possible transformations!')
        self.tree_lstm_cell = BinaryTreeLstmCell(hidden_dim, dropout_prob)
        BinaryTreeBasedModule.reset_parameters(self)

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)
        self.tree_lstm_cell.reset_parameters()
        if self.leaf_transformation == BinaryTreeBasedModule.lstm_transformation:
            self.lstm.reset_parameters()
        elif self.leaf_transformation == BinaryTreeBasedModule.bi_lstm_transformation:
            self.lstm_f.reset_parameters()
            self.lstm_b.reset_parameters()
        elif self.leaf_transformation == BinaryTreeBasedModule.conv_transformation:
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()

    def forward(self, *inputs):
        raise NotImplementedError

    def _transform_leafs(self, x, mask):
        """ Transforms token embeddings to the correct shape for the TreeLSTM. 
        
        The transformation depends on the type of leaf transformation used in the model.
        The possible transformations are:
        - no_transformation: no transformation is applied, the input is used as is.
        - lstm_transformation: a single LSTM is applied to the input.
        - bi_lstm_transformation: a bidirectional LSTM is applied to the input.
        - conv_transformation: a convolutional layer is applied to the input.

        The input is expected to be of shape (batch_size, seq_len, input_dim), where input_dim is the dimension of the token embeddings.
        There are two outputs: the hidden state (h) and the cell state (c) of the LSTM.
        
        """
        if self.leaf_transformation == BinaryTreeBasedModule.no_transformation:
            pass
        elif self.leaf_transformation == BinaryTreeBasedModule.lstm_transformation:
            x = self.lstm(x, mask)
        elif self.leaf_transformation == BinaryTreeBasedModule.bi_lstm_transformation:
            h_f = self.lstm_f(x, mask)
            h_b = self.lstm_b(x, mask, backward=True)
            x = torch.cat([h_f, h_b], dim=-1)
        elif self.leaf_transformation == BinaryTreeBasedModule.conv_transformation:
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.tanh(x)
            x = x.permute(0, 2, 1)

        # Apply linear projection to the transformed input to get the hidden and cell states
        # the output is split arbitrarily into two parts, which are used as the hidden and cell states of the TreeLSTM
        # the model then learns to use them as it sees fit
        return self.linear(x).tanh().chunk(chunks=2, dim=-1)

    @staticmethod
    def _merge(actions, h_l, c_l, h_r, c_r, h_p, c_p, mask):
        """
        This method merges left and right TreeLSTM states. It reuses already precomputed states for the parent node,
        but still, has to apply correct masking.
        """
        # create masks for left and right children
        cumsum = torch.cumsum(actions, dim=-1)
        mask_l = (1.0 - cumsum)[..., None]      # 1 before the merge
        mask_r = (cumsum - actions)[..., None]  # 1 after the merge
        mask = mask[..., None]                  # 1 for non-padding spans
        actions = actions[..., None]            # 1 for the current merge action

        # new states combine the old states multiplied by the masks
        h_p = (mask_l * h_l + actions * h_p + mask_r * h_r) * mask + h_l * (1. - mask)
        c_p = (mask_l * c_l + actions * c_p + mask_r * c_r) * mask + c_l * (1. - mask)
        return h_p, c_p
