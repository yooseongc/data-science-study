


from typing import List, NamedTuple, Dict
import math, random
from collections import Counter, defaultdict
from linear_algebra import Vector, distance
import matplotlib.pyplot as plt



"""
k-NN (k-Nearest Neighbors, k-근접이웃)

 * 서울특별시장 선거를 생각해보자.
 * 내 주변 이웃들이 누구를 뽑을지를 알고 있다면, 내가 누구를 뽑을지 짐작해 볼 수 있음.
 * 왜냐하면 나의 행동은 주변에 사는 사람들의 영향을 받을 수 있기 때문.
 * 지역 뿐 아니라 나이, 소득수준, 자녀 수 등에 대해서도 같은 방법을 사용한다면,
   나는 그런 특성들에 의해 어느정도 영향을 받을 것이기 때문에,
   이를 이용해 내 특성에 좀 더 가까운 이웃을 선별한다면
   모든 이웃을 고려했을 때보다 더 나은 추정을 할 수 있을 것임. 

 * k-NN은 가장 단순한 예측 모델 중 하나임.
 * 수학적인 가정도 필요없고, 엄청난 컴퓨팅 파워가 필요한 것도 아님.
 * 필요한 것은
    - 거리를 재는 방법
    - 서로 가까운 점들은 유사하다는 가정
   이 두가지임.
 * 다른 머신러닝 기술들은 대체로 내재된 패턴을 찾기 위해 데이터셋 전체를 탐색하지만,
   k-NN은 주변만 보면 되기 때문에 데이터 탐색량이 많지 않음.
 * 다만, 특정 현상의 원인을 파악하는 데에는 적절하지 않음.
   가령, 의사결정트리는 판가름이 되는 기준을 얘기해 주지만, k-NN은 그렇지 않음.

 * 보통 데이터 포인트 몇 개와 그에 대한 label 정보가 주어짐.
 * label은 '스팸인가?', '독성이 있는가?', '재미있는가?'와 같이 
   특정 조건을 만족했는지에 따라 참/거짓이 되거나
   영화 등급과 같이 다양한 카테고리가 될 수 있음.
   또는 대선 후보 이름이나 좋아하는 프로그래밍 언어일 수도 있음.
 * 각 데이터 포인트는 벡터로 표현할 수 있으며, 벡터의 distance를 계산하여 거리를 구할 수 있음.

 * 먼저 k를 3 또는 5로 정했다고 해보자.
 * 새로운 데이터 포인트를 분류하고 싶다면, 먼저 k개의 가장 가까운 포인트를 찾고,
   찾아낸 포인트들의 레이블을 보고 다수결(majority vote)로 새로운 데이터 포인트의 레이블을 정할 수 있음.
 * 이제 다수결을 정하는 함수를 만들어보자.

"""

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]  # label, number of votes
    return winner

assert raw_majority_vote(['a', 'b', 'c', 'b']) == 'b'

"""
 * 이 방법은 동점인 항목들이 제대로 처리되지 않음. 
 * 만약 공동1등이라면 어떻게 1등을 추려야 할까?
 * 아래와 같은 방법이 있음.
   - 여러 1등 중 임의로 하나를 정함.
   - 거리를 가중치로 사용하여 거리 기반 투표를 함.
   - 단독 1등이 생길 때 까지 k를 하나씩 줄임.
 * 아래는 세번째 방법의 구현 (labels가 거리 순서로 정렬되어 있다고 가정함.)
   
"""

def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])  # try again without the farthest

# Tie, so look at first 4, then 'b'
assert majority_vote(['a', 'b', 'c', 'b', 'a']) == 'b'


class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:

    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points, key = lambda lp: distance(lp.point, new_point))

    # Find the labels for the k-closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # and let them vote.
    return majority_vote(k_nearest_labels)


cities = [(-86.75,33.5666666666667,'Python'),(-88.25,30.6833333333333,'Python'),(-112.016666666667,33.4333333333333,'Java'),(-110.933333333333,32.1166666666667,'Java'),(-92.2333333333333,34.7333333333333,'R'),(-121.95,37.7,'R'),(-118.15,33.8166666666667,'Python'),(-118.233333333333,34.05,'Java'),(-122.316666666667,37.8166666666667,'R'),(-117.6,34.05,'Python'),(-116.533333333333,33.8166666666667,'Python'),(-121.5,38.5166666666667,'R'),(-117.166666666667,32.7333333333333,'R'),(-122.383333333333,37.6166666666667,'R'),(-121.933333333333,37.3666666666667,'R'),(-122.016666666667,36.9833333333333,'Python'),(-104.716666666667,38.8166666666667,'Python'),(-104.866666666667,39.75,'Python'),(-72.65,41.7333333333333,'R'),(-75.6,39.6666666666667,'Python'),(-77.0333333333333,38.85,'Python'),(-80.2666666666667,25.8,'Java'),(-81.3833333333333,28.55,'Java'),(-82.5333333333333,27.9666666666667,'Java'),(-84.4333333333333,33.65,'Python'),(-116.216666666667,43.5666666666667,'Python'),(-87.75,41.7833333333333,'Java'),(-86.2833333333333,39.7333333333333,'Java'),(-93.65,41.5333333333333,'Java'),(-97.4166666666667,37.65,'Java'),(-85.7333333333333,38.1833333333333,'Python'),(-90.25,29.9833333333333,'Java'),(-70.3166666666667,43.65,'R'),(-76.6666666666667,39.1833333333333,'R'),(-71.0333333333333,42.3666666666667,'R'),(-72.5333333333333,42.2,'R'),(-83.0166666666667,42.4166666666667,'Python'),(-84.6,42.7833333333333,'Python'),(-93.2166666666667,44.8833333333333,'Python'),(-90.0833333333333,32.3166666666667,'Java'),(-94.5833333333333,39.1166666666667,'Java'),(-90.3833333333333,38.75,'Python'),(-108.533333333333,45.8,'Python'),(-95.9,41.3,'Python'),(-115.166666666667,36.0833333333333,'Java'),(-71.4333333333333,42.9333333333333,'R'),(-74.1666666666667,40.7,'R'),(-106.616666666667,35.05,'Python'),(-78.7333333333333,42.9333333333333,'R'),(-73.9666666666667,40.7833333333333,'R'),(-80.9333333333333,35.2166666666667,'Python'),(-78.7833333333333,35.8666666666667,'Python'),(-100.75,46.7666666666667,'Java'),(-84.5166666666667,39.15,'Java'),(-81.85,41.4,'Java'),(-82.8833333333333,40,'Java'),(-97.6,35.4,'Python'),(-122.666666666667,45.5333333333333,'Python'),(-75.25,39.8833333333333,'Python'),(-80.2166666666667,40.5,'Python'),(-71.4333333333333,41.7333333333333,'R'),(-81.1166666666667,33.95,'R'),(-96.7333333333333,43.5666666666667,'Python'),(-90,35.05,'R'),(-86.6833333333333,36.1166666666667,'R'),(-97.7,30.3,'Python'),(-96.85,32.85,'Java'),(-95.35,29.9666666666667,'Java'),(-98.4666666666667,29.5333333333333,'Java'),(-111.966666666667,40.7666666666667,'Python'),(-73.15,44.4666666666667,'R'),(-77.3333333333333,37.5,'Python'),(-122.3,47.5333333333333,'Python'),(-89.3333333333333,43.1333333333333,'R'),(-104.816666666667,41.15,'Java')]
cities = [LabeledPoint([longitude, latitude], language) for longitude, latitude, language in cities]

for longitude in range(-130, -60):
    for latitude in range(20, 55):
        predicted_language = knn_classify(1, cities, [longitude, latitude])
        print(longitude, latitude, predicted_language)


def main():
    from matplotlib import pyplot as plt
    import csv


if __name__ == "__main__":
    main()
