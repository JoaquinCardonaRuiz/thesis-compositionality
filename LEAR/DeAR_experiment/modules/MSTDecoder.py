import numpy as np
def mst_decode(arc_scores, root_index=0):
        """
        Simple MST decoding for dependency arcs.

        Args:
            arc_scores: Tensor (seq_len, seq_len), score[i][j] = i → j
            root_index: Index of artificial ROOT token (default 0)

        Returns:
            heads: list[int] of length seq_len, where heads[j] = i means i → j
        """

        L = arc_scores.size(0)
        scores = arc_scores.detach().cpu().numpy()

        heads = np.full(L, -1)
        heads[root_index] = root_index  # root self-loop

        for dep in range(L):
            if dep == root_index:
                continue
            heads[dep] = np.argmax(scores[:, dep])

        return list(heads)