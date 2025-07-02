import numpy as np

def initialize_intervals(values, labels):
    sorted_indices = np.argsort(values)
    sorted_values = np.array(values)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    
    intervals = []
    for v, lbl in zip(sorted_values, sorted_labels):
        # Konvertiere NumPy-Array in Tupel (wenn Array) oder String
        if isinstance(lbl, np.ndarray):
            key = tuple(lbl)  # Konvertiere Array in hashbares Tupel
        else:
            key = str(lbl)    # Konvertiere Skalar in String
        
        intervals.append({
            'start': v,
            'end': v,
            'count': {key: 1}  # Verwende konvertierten Schlüssel
        })
    return intervals


def merge_intervals(interval1, interval2):
    # Fasse zwei Intervalle zusammen
    merged = {
        'start': interval1['start'],
        'end': interval2['end'],
        'count': {}
    }
    all_labels = set(interval1['count'].keys()).union(interval2['count'].keys())
    for label in all_labels:
        merged['count'][label] = interval1['count'].get(label, 0) + interval2['count'].get(label, 0)
    return merged

def compute_chi2(interval1, interval2, all_labels):
    # Erstelle Kontingenztabelle
    obs = np.zeros((2, len(all_labels)))
    for idx, label in enumerate(all_labels):
        obs[0, idx] = interval1['count'].get(label, 0)
        obs[1, idx] = interval2['count'].get(label, 0)
    # Summen berechnen
    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)
    total = obs.sum()
    # Erwartete Häufigkeiten
    expected = np.outer(row_sums, col_sums) / total
    # Chi2 berechnen (nur für erwartete Werte > 0)
    mask = expected > 0
    chi2_val = ((obs - expected) ** 2 / expected)[mask].sum()
    return chi2_val
