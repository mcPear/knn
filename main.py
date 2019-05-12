from sklearn import neighbors
from util.validation import k_fold
import util.datasets as datasets
from util.utils import standarise

vote_uniform = 'uniform'
vote_distance_weighted = 'distance'
euclidean_metric = 'euclidean'
manhattan_metric = 'manhattan'

k = 5
vote = vote_uniform
metric = euclidean_metric
folds = 10
data = datasets.iris()
standarise(data)

for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights=vote, metric=metric, algorithm='brute', n_jobs=-1)
    
    f1 = k_fold(data, knn, folds, True)

    print(f1)
