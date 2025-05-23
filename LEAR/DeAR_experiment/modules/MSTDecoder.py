import networkx as nx

def mst_decode(arc_scores):
    """
    Chu-Liu/Edmonds MST decoding using networkx.
    """
    L = arc_scores.size(0)
    scores = arc_scores.detach().cpu().numpy()

    g = nx.DiGraph()
    for i in range(L):
        for j in range(L):
            if i == j:
                continue  # still skip self-loops
            g.add_edge(i, j, weight=scores[i][j])

    mst = nx.maximum_spanning_arborescence(g, preserve_attrs=True)

    heads = [-1] * L
    for u, v in mst.edges:
        heads[v] = u

    return heads