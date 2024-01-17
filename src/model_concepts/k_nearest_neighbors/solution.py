import math
from typing import List, TypedDict, Dict


class Feature(TypedDict):
    features: List[float]
    is_intrusive: int


def predict_label(
    examples: Dict[str, Feature],
    features: List[float],
    k: int,
    label_key: str = "is_intrusive",
) -> int:
    k_nearest_neighbors = find_k_nearest_neighbors(examples, features, k)
    k_nearest_neighbors_labels: List[int] = [
        examples[pid][label_key] for pid in k_nearest_neighbors
    ]
    return round(sum(k_nearest_neighbors_labels) / k)


def find_k_nearest_neighbors(
    examples: Dict[str, Feature], features: List[float], k: int
) -> List[str]:
    distances: Dict[str, float] = {}
    for pid, features_label_map in examples.items():
        distance = get_euclidean_distance(features, features_label_map["features"])
        distances[pid] = distance
    return sorted(distances, key=distances.get)[:k]


def get_euclidean_distance(features: List[float], other_features: List[float]) -> float:
    squared_differences = []
    for i in range(len(features)):
        squared_differences.append((other_features[i] - features[i]) ** 2)
    return math.sqrt(sum(squared_differences))
