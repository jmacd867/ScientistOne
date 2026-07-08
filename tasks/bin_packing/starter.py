def pack(items: list[float], capacity: float) -> list[list[float]]:
    """Pack items into bins of the given capacity. Return a list of bins.

    Baseline: naive first-fit. Improve on this.
    """
    bins: list[list[float]] = []
    for item in items:
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                break
        else:
            bins.append([item])
    return bins
