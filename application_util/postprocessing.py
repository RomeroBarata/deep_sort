import operator

import numpy as np


def filter_top_k_longest_tracks(results, k, criterion='length'):
    min_id, max_id = int(np.min(results[:, 1]).item()), int(np.max(results[:, 1]).item())
    track_id_to_track_quality = {}
    if criterion == 'length':
        for track_id in range(min_id, max_id + 1):
            track_id_to_track_quality[track_id] = np.sum(results[:, 1] == track_id)
    elif criterion == 'accumulated_score':
        for track_id in range(min_id, max_id + 1):
            mask = results[:, 1] == track_id
            track_id_to_track_quality[track_id] = np.sum(results[mask, 3])
    else:
        raise ValueError('Unknown criterion %s' % criterion)
    sorted_data = sorted(track_id_to_track_quality.items(), key=operator.itemgetter(1))  # (track_id, track_len)
    top_k_data = sorted_data[-k:]
    top_k_ids = {top_k_id for top_k_id, _ in top_k_data}
    results = np.stack([result for result in results if result[1] in top_k_ids], axis=0)
    return results
