from scipy.cluster.hierarchy import linkage, fcluster


def build_linkage(X, method="ward"):
    """Build the full merge tree. Z has one row per merge."""
    return linkage(X, method=method)


def cut_tree(Z, k):
    """Cut the tree into k clusters. Labels start at 0, like K-Means."""
    return fcluster(Z, t=k, criterion="maxclust") - 1


def merge_heights(Z, n=10):
    """Heights of the last n merges, starting with the final merge."""
    return Z[-n:, 2][::-1]