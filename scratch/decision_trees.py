#!/usr/bin/env python

from typing import List, Any, Union, NamedTuple, Optional, Dict, TypeVar
import math
from collections import Counter, defaultdict

"""
 * 의사결정나무(decision trees)는 다양한 의사결정 경로(decision path)와 결과(outcome)를 나타내는 데 나무 구조를 사용한다.
 * 스무고개 놀이를 해 본 사람이라면 의사결정나무에 익숙할 것이다.
 * 간단한 스무고개 예시 : '다리가 다섯 개 이상? -> NO' > '맛있음? -> NO' > '호주의 5센트 동전 뒷면에 등장함? YES' 

 * 장점
   - 이해하고 해석하기가 쉬움
   - 프로세스가 명쾌
   - 숫자형 데이터와 범주형 데이터를 동시에 다룰 수 있으며, 특정 변수 값이 누락되어도 사용 가능

 * 단점
   - '최적'의 의사결정나무를 찾는 것은 매우 어려움
   - '오버피팅'되기 쉬움

 * 분류
   - 분류나무(classification tree) : 범주형 결과 반환 
   - 회귀나무(regression tree)     : 숫자형 결과 반환

 * 의사결정나무를 만들기 위해 어떤 질문을 묻고, 어떤 순서로 질문을 할 것인가를 정해야 함.
 * 모든 질문은 답이 무엇인지에 따라 잔존하는 옵션을 분류시킴.
 * 이상적으로는, 예측하려는 대상에 대해 가장 많은 정보를 담고 있는 질문을 고르는 것이 좋음. 
 * 질문을 던져도 결과값에 대한 새로운 정보를 전혀 주지 못하는 질문은 좋은 질문이 아님.

 * Entropy : 얼마만큼의 정보를 담고 있는가에 대한 정량적 수치
 * 데이터셋 S와 c_1, c_2, ... , c_n 등 유한 개의 클래스가 있다고 하자.
   각 데이터 포인트는 c_1 ... c_n 중 하나에 속한다고 하자.
   만약 모든 데이터 포인트가 각각 단 하나의 클래스에 속한다면 불확실성은 전혀 없고 엔트로피는 낮을 것임.
   반면 모든 데이터 포인트가 모든 클래스에 고르게 분포되어 있다면 불확실성은 매우 높고 엔트로피도 높을 것임.
   한 데이터 포인트가 클래스 c_i에 속할 확를을 p_i라 한다면, 엔트로피 H는
   H(S) = - p_1 * log_2(p_1) - p_2 * log_2(p_2) - ... - p_n * log_2(p_n) 으로 표기할 수 있음.
   이진로그를 사용하는 이유는 정보의 단위가 bit니까.
   단, 0 * log_2(0) = 0, 각 항 - p_i * log_2(p_i)의 값은 항상 0보다 크거나 같으며 p_i의 값이 0 또는 1에 가까울 수록 0에 수렴함.
 * 모든 p_i가 0 또는 1에 가까우면 엔트로피는 아주 작을 것이고, 그렇지 않다면 (데이터 포인트들이 여러 클래스에 고르게 분포한다면) 엔트로피는 큰 값을 가짐.

"""

def entropy(class_probabilites: List[float]) -> float:
    """Given a list of class probabilities, compute the entropy"""
    return sum(- p * math.log(p, 2) for p in class_probabilites if p)  # 확률이 0인 경우는 제외, 1인 경우는 log의 성질에 의해 알아서 0이 됨.

assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82

"""
 * 입력 데이터는 (input, label) 쌍으로 구성되어 있기 때문에, 각 클래스 레이블의 확률은 별도의 계산이 필요.
 * 레이블과 무관하게 확률 값들만 알면 됨.
"""

def class_probabilites(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]  # 라벨 클래스의 빈도 수를 이용해 확률을 계산

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilites(labels))

assert data_entropy(['a']) == 0                              # 'a' : 1.0
assert data_entropy([True, False]) == 1                      # True : 0.5, False: 0.5
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])   # 3 : 0.25, 4: 0.75


"""
 * 의사결정나무의 각 단계는 데이터를 데이터셋 전체가 아닌 여러 개의 파티션(partition)으로 분할함.
 * 가령, '다리가 다섯 개 이상인가?' 라는 질문에 대해서 YES에 해당하는 동물과 NO에 해당하는 동물의 파티션으로 나뉠 것임.
 * 여러 개의 파티션으로 나뉘더라도 데이터셋 전체에 대한 엔트로피를 계산할 수 있는 방법이 필요함.
 * 앞의 스무고개 질문 중 '호주 5센트' 질문은 남은 동물의 집합을 바늘두더지(S_1)와 다른 동물(S_2)로 나누어버리기 때문에 멍청한 질문이라 할 수 있음.
 * S_2는 집합도 크고 엔트로피도 높은 반면, S1은 엔트로피가 0이며 집합도 작음.
 * S를 q_1, q_2, ... , q_m의 비율을 가지는 파티션 S_1, S_2, ... , S_m 으로 나누는 경우, 엔트로피는
   H = q_1 * H(S_1) + ... + q_m * H(S_m) 으로 계산함.
"""

def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)


"""
 부사장이 (input, label) 쌍으로 구성된 인터뷰 후보자의 데이터를 전달해 주었다.
 이때 input에는 후보자의 다양한 속성이 dict의 형태로 담겨 있고, 
 label에는 인터뷰를 잘 했음(True) 또는 잘 못 했음(False)의 값이 있음.
 후보자에 대한 변수는 직급(level), 선호하는 프로그래밍 언어(lang), 트위터 사용 여부(tweets), 박사 학위 유무(phd) 등 네 가지가 주어짐.
"""

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data

                  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]

T = TypeVar('T')  # generic type for inputs

"""
 * 의사결정나무는 결정 노드(decision node)와 잎 노드(leaf node)로 구성됨.
 * 결정 노드는 질문을 주고 답변에 따라 다른 경로로 안내를 해 줌.
 * 잎 노드는 예측 값이 무엇인지 알려줌.
 * tree는 ID3 (Iterative Dichotomiser 3) 알고리즘에 기반하여 구축할 수 있음. 
 * https://en.wikipedia.org/wiki/ID3_algorithm

 * 위 input과 같이 label(did_well)과 partition을 나눌 수 있는 변수가 주어졌다고 해 보면,
 * 아래와 같은 알고리즘을 진행함.
   1) 모든 데이터 포인트의 클래스 레이블이 동일하다면, 그 예측값이 해당 클래스 레이블인 잎 노드를 만들고 종료
   2) 파티션을 나눌 수 있는 변수가 남아 있지 않다면(더 이상 물을 수 있는 질문이 없다면), 가장 빈도 수가 높은 클래스 레이블로 예측하는 잎 노드를 만들고 종료
   3) 1, 2에 해당하지 않는다면 각 변수로 데이터의 파티션을 나눔
   4) 파티션을 나눴을 때 엔트로피가 가장 낮은 변수를 택함
   5) 선택된 변수에 대한 결정 노드를 추가함
   6) 남아 있는 변수들로 각 파티션에 대해 위 과정을 반복함

   => greedy(탐욕적) 알고리즘으로, 순간순간에 가장 최적이라고 생각되는 선택을 함
      단, 전체로 봤을 때 더 좋은 선택인 경우지만 순간 가장 최적이 아니라면 건너뛰게 됨

 * 아래 과정을 통해 위 단계를 실행해 보자.
 * 네 가지 변수로 나무의 가지를 나눌 수 있음. 
"""

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attributes."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)
        partitions[key].append(input)
    return partitions

import pprint
pp = pprint.PrettyPrinter(indent=2)

def partition_entropy_by(inputs: List[Any], attribute: str, label_attribute: str) -> float:
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs
    partitions = partition_by(inputs, attribute)
    
    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition] for partition in partitions.values()]
    #pp.pprint(partitions)
    #pp.pprint(labels)
    return partition_entropy(labels)

# for key in ['level', 'lang', 'tweets', 'phd']:
#     print(key, partition_entropy_by(inputs, key, 'did_well'))

assert 0.69 < partition_entropy_by(inputs, 'level', 'did_well')  < 0.70 
assert 0.86 < partition_entropy_by(inputs, 'lang', 'did_well')   < 0.87
assert 0.78 < partition_entropy_by(inputs, 'tweets', 'did_well') < 0.79
assert 0.89 < partition_entropy_by(inputs, 'phd', 'did_well')    < 0.90

# level 0.6935361388961919
# lang 0.8601317128547441
# tweets 0.7884504573082896
# phd 0.8921589282623617

"""
 * 위 결과에 따라 엔트로피가 최소가 되는 attribute인 level(직급)을 결정노드로 하고 subtree를 만들자.
 * 직급이 Mid인 경우에는 모든 레이블(did_well)이 True이므로 이 subtree는 leaf node가 될 것임.
 * level이 Senior인 경우, tweets에 대해 레이블이 완전히 나뉘게 되므로 tweets가 다음 결정노드, 그 하위로 두 개의 leaf node가 생김.
"""

senior_inputs = [input for input in inputs if input.level == 'Senior']
assert 0.4 == partition_entropy_by(senior_inputs, 'lang', 'did_well')
assert 0.0 == partition_entropy_by(senior_inputs, 'tweets', 'did_well')
assert 0.95 < partition_entropy_by(senior_inputs, 'phd', 'did_well') < 0.96


"""
 * 위 내용을 종합화여 일반화시켜 보자.
 * 가장 먼저 해야 할 일은 나무를 어떻게 표현할지 결정하는 것
 * 나무가 True, False, (변수, 서브트리(dict))로 구성된 tuple의 세 가지 중 하나의 값을 반드시 가져야 한다고 하자.
 * True: 어떤 값을 입력받아도 True를 반환하는 잎 노드
 * False: 어떤 값을 입력받아도 False를 반환하는 잎 노드
 * tuple: 변수와 그것으로 입력값을 분류한 서브트리로 구성된 결정 노드
 * 이 표현법에 따라 의사결정나무는 아래의 형태를 따름
   ('level', 
        {
            'Junior': ('phd', { 'no': True, 'yes': False }),
            'Mid': True
            'Senior': ('tweets', { 'no': False, 'yes': True })
        }   
   )
 * 한편, 새로운 후보자가 왔을 때 해당 후보자에 대한 변수 값 중 하나가 기존에 관찰되지 않은 것이면 어떻게 하면 좋을까?
 * 가령 새로 입력 받은 후보자의 level이 'Intern'이라 하면 None이라는 key 값을 추가해서 가장 빈도가 높은 클래스 레이블을 
   할당할 수 있을 것임.
 * 본격적으로 구현해보자.
"""

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

hiring_tree = Split('level', {
    'Junior': Split('phd', { False: Leaf(True), True: Leaf(False) }),
    'Mid': Leaf(True),
    'Senior': Split('tweets', { False: Leaf(False), True: Leaf(True) })
})

def classify(tree: DecisionTree, input: Any) -> Any:
    """classify the input using the given decision tree"""

    # 1) If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value
    
    # 2) Otherwise this tree consists of an attribute to split on
    #      and a dictionary whose keys are values of that attribute
    #      and whose values of are subtrees to consider next
    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees:      # If no subtree for key,
        return tree.default_value             # return the default value.
    
    subtree = tree.subtrees[subtree_key]      # Choose the appropriate subtree
    return classify(subtree, input)           # and use it to classify the input.

def build_tree_id3(inputs: List[Any], split_attributes: List[str], target_attribute: str) -> DecisionTree:
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    # If there's a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)
    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)
    # Otherwise split by the best attribute
    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partition_entropy_by(inputs, attribute, target_attribute)
    
    best_attribute = min(split_attributes, key=split_entropy)
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]
    
    # recursively build the subtree
    subtrees = { attribute_value: build_tree_id3(subset, new_attributes, target_attribute) 
                    for attribute_value, subset in partitions.items() }
    return Split(best_attribute, subtrees, default_value=most_common_label)

tree = build_tree_id3(inputs, ['level', 'lang', 'tweets', 'phd'], 'did_well')


assert classify(tree, Candidate("Junior", "Java", True, False))     # True
assert not classify(tree, Candidate("Junior", "Java", True, True))  # False
assert classify(tree, Candidate("Intern", "Java", True, True))      # True (most common label of "Intern")
assert classify(tree, Candidate(None, "Java", True, True))          # True (most common label of None)

"""
 * label 없는 임의의 테스트 데이터도 분류가 가능.
 * 관찰된 적 없는 값이 변수(attribute)에 등장하거나 변수 값 자체가 누락 된 경우에도 분류가 가능.
 * 실제로 모델을 구축한다면, 더 많은 데이터를 수집한 후 데이터를 학습, 검증, 평가 데이터로 나눠서 진행을 할 것임.
"""

"""
 * 오버피팅을 방지할 수 있는 대표적인 방법으로 'Random Forests'가 있음.
 * 여러 개의 의사결정나무를 만들고, 그들의 다수결로 결과를 결정하는 방법임.
 * greedy한 부분을 해결하는 방법이라 이해해도 될 듯.
"""

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

"""
 * 지금까지 구축한 나무에는 랜덤성이 없음.
 
 * 랜덤한 나무를 얻기 위한 한 가지 방법은, 데이터를 bootstrap하는 것.
 * inputs 전체를 이용하는 것이 아닌, bootstrap_sample(inputs)와 같이 일부를 샘플링하여 나무를 여러 개 만들면,
   각 나무는 랜덤성을 지니게 됨.
 * 부가적으로 샘플링 되지 않은 데이터는 테스트 데이터로 이용 가능.
 * 이와 같은 방법을 'bootstrap aggregating' 또는 'bagging'이라 함.

 * 두 번째 방법은 파티션을 나누는 변수에 랜덤성을 부여하는 것.
 * 즉 남아있는 모든 변수 중 최적의 변수를 선택하는 것이 아니라, 변수 중 일부만 선택하고 그 일부 중에서 최적의 변수를 선택.
 * 이런 방법을 광범위하게 'ensemble learning'이라고 하는데, 이는 성능이 떨어지는(대체로 bias가 높고 variance가 낮은) 여러 모델을
   동시에 활용해서 전체적으로 성능이 좋은 모델을 구축하는 것.
"""

# Tree Building 코드를 아래와 같이 수정하면 두 번째 방법대로 Random Forests를 만들 수 있음.
#
# if len(split_attributes) <= self.num_split_attributes:  # 사용할 변수가 얼마 남지 않았다면 전부 사용함
#     sampled_split_attributes = split_attributes
# else:                                                   # 그렇지 않다면 사용할 변수를 랜덤으로 선택
#     sampled_split_attributes = random.sample(split_attributes, self.num_split_attributes)

# # 선택된 변수 중 가장 적절한 변수를 선택
# best_attribute = min(sampled_split_attributes, key=partial(partition_entropy_by, inputs))
# partitions = partition_by(inputs, best_attribute)


"""
 * scikit-learn에는 많은 의사결정나무 모델이 구현되어 있으며, ensemble module을 비롯해 RandomForestClassifier가 포함되어 있음.
 * Wikipedia를 시작으로 더 공부해 봐도 좋을 것임.
 * https://en.wikipedia.org/wiki/Decision_tree_learning
 * https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95_%ED%8A%B8%EB%A6%AC_%ED%95%99%EC%8A%B5%EB%B2%95

 * 의사 결정 분석에서 결정 트리는 시각적이고 명시적인 방법으로 의사 결정 과정과 결정된 의사를 보여주는데 사용
 * 데이터 마이닝 분야에서 결정 트리는 결정된 의사보다는 자료 자체를 표현하는데 사용
 * 데이터 마이닝의 결과로서의 분류 트리는 의사 결정 분석의 입력 값으로 사용될 수 있음
 * 결정 트리 학습법은 데이터 마이닝에서 일반적으로 사용되는 방법론으로, 몇몇 입력 변수를 바탕으로 목표 변수의 값을 예측하는 모델을 생성하는 것이 목표
 * 결정 트리 학습법은 지도 분류 학습에서 가장 유용하게 사용되고 있는 기법 중 하나

 * 분류 트리 분석은 예측된 결과로 입력 데이터가 분류되는 클래스를 출력
 * 회귀 트리 분석은 예측된 결과로 특정 의미를 지니는 실수 값을 출력 (예: 주택의 가격, 환자의 입원 기간)
 * 회귀 및 분석 트리(Classification And Regression Tree, CART)는 두 트리를 아울러 일컫는 용어로, 레오 브레이만에 의해 처음 사용
 * 회귀 트리와 분류 트리는 일정 부분 유사하지만, 입력 자료를 나누는 과정 등에서 차이점이 있음

 * 앙상블 방법
  - 배깅 방법 (초기 앙상블 방법)
  - 랜덤 포레스트 분류기에서는 분류 속도를 향상시키기 위해서 결정 트리들을 사용
  - 부스트 트리는 회귀 분석과 분류 문제에 사용
  - 회전 포레스트는 모든 결정 트리가 먼저 입력 트리 중 임의의 부분 집합에 대한 주성분 분석 (PCA)을 적용하여 훈련

 * 알고리즘
  - ID3 (Iterative Dichotomiser 3)
  - C4.5 (successor of ID3)
  - C5.0 (successor of ID4)
  - CART (Classification And Regression Tree)
  - CHAID (CHi-squared Automatic Interaction Detector) : 이 알고리즘은 분류 트리를 계산할 때 다단계 분할을 수행한다.
  - MARS (Multivariate adaptive regression splines) : 더 많은 수치 데이터를 처리하기 위해 결정 트리를 사용한다.
  - 조건부 추론 트리 (Conditional Inference Trees) : 과적합을 피하기 위해 여러 테스트에 대해 보정 분할 기준으로 비-파라미터 테스트를 사용하는 통계 기반의 방법이다. 이 방법은 편견 예측 선택 결과와 가지치기가 필요하지 않다.
"""
