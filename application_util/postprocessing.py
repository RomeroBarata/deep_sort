import operator

import numpy as np


def filter_top_k_longest_tracks(results, k):
    min_id, max_id = int(np.min(results[:, 1]).item()), int(np.max(results[:, 1]).item())
    track_id_to_track_len = {}
    for track_id in range(min_id, max_id + 1):
        track_id_to_track_len[track_id] = np.sum(results[:, 1] == track_id)
    sorted_data = sorted(track_id_to_track_len.items(), key=operator.itemgetter(1))  # (track_id, track_len)
    top_k_data = sorted_data[-k:]
    top_k_ids = {top_k_id for top_k_id, _ in top_k_data}
    results = np.stack([result for result in results if result[1] in top_k_ids], axis=0)
    return results
