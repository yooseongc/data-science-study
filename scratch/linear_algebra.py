#!/usr/bin/env python

from typing import List, Callable, Tuple
import math

"""

 * 선형대수 : 벡터 공간을 다루는 수학의 한 분야

 * 벡터(Vector) : 벡터끼리 더하거나 상수와 곱해지면 새로운 벡터를 생성하는 개념적인 도구,
                 유한한 차원의 공간에 존재하는 점들.
 * 대부분의 데이터, 특히 숫자로 표현된 데이터는 벡터로 표현이 가능함.
 * 가장 간단하게 표현하는 방법은 숫자로 구성된 list로 표현하는 것임.
 * 단, Python의 list는 벡터가 아니므로, 벡터 연산을 해주는 기본적인 도구가 필요함.
 * 아래에서 직접 구현을 해 보자.
 * 단, 실제로 계산할 때는 NumPy 라이브러리 사용을 강력하게 권장함.
 * Python의 List는 끔찍한 성능을 보임.

"""

Vector = List[float]

height_weight_age = [70, 170, 40]  # human properties - [ inches, pounds, years ]
grades = [95, 80, 75, 62]          # test score - [ exam1, exam2, exam3, exam4 ]

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    assert vectors, "no vectors provided!"
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    """Computes v_1 * v_1 + v_2 * v_2 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v: Vector) -> float:
    """Computes the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

# def distance(v: Vector, w: Vector) -> float:
#     """Computes the distance between v and w"""
#     return math.sqrt(squared_distance(v, w))

def distance(v: Vector, w: Vector) -> float:  # type: ignore
    return magnitude(subtract(v, w))

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6
assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3
assert magnitude([3, 4]) == 5


"""

 * 행렬(Matrix) : 2차원으로 구성된 숫자의 집합이며, list의 list로 표현 가능.
 * list안의 list들은 행렬의 행을 나타내며, 모두 같은 길이를 가짐.
 * A[i][j]는 A라는 행렬의 i번째 행과 j번째 열에 속한 숫자를 의미함.
 A = [[1, 2, 3], [4, 5, 6]]
 B = [[1, 2], [3, 4], [5, 6]]

 A =  | 1  2  3 |
      | 4  5  6 |

 B = | 1  2 |
     | 3  4 |
     | 5  6 |
     
 * 아래에서 구현해 보자. 
"""

Matrix = List[List[float]]

A = [[1, 2, 3], [4, 5, 6]]
B = [[1, 2], [3, 4], [5, 6]]

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j] for A_i in A]

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

def is_diagonal(i: int, j: int) -> int:
    return 1 if i == j else 0

def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, is_diagonal)



assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 rows, 3 columns
assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]


"""
 * 행렬은 아래와 같은 이유로 매우 중요함.
   - 각 벡터를 행렬의 행으로 (혹은 열로) 나타냄으로써 여러 벡터로 구성된 데이터셋을 행렬로 표현 가능함.
        가령, 1000명에 대한 키, 몸무게, 나이가 주어졌다면 1000x3 행렬로 표현 가능.
   - k차원의 벡터를 n차원 벡터로 변환해 주는 선형함수를 n x k로 표현할 수 있음.
   - 행렬로 이진 관계 (binary relationship)을 나타낼 수 있음. 
        가령, 노드와 노드의 연결 구조를 그래프 행렬로 나타낼 수 있음.  
"""

friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # user 0
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # user 1
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # user 2
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # user 3
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # user 4
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # user 5
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 6
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # user 7
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # user 8
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]  # user 9

friends_of_five = [i for i, is_friend in enumerate(friend_matrix[5]) if is_friend]

assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are not friends"
assert friends_of_five == [4, 6, 7]

