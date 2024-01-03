import random

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_array, check_random_state
from scipy.spatial import distance_matrix

__author__ = 'taoll'

class HSNLO(object):

    def __init__(self,
                 k_min = 9,
                 k_maj = 2,
                 max_weight_percentile=90,
                 p_norm=2,
                 verbose=False):
        self.k_min = k_min
        self.k_maj = k_maj
        self.max_weight_percentile = max_weight_percentile
        self.p_norm = p_norm
        self.verbose = verbose

    def fit(self,X,y):

        self.X = check_array(X)
        self.y = np.array(y)

        classes = np.unique(y)

        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        self.unique_classes_ = classes[indices]
       
        self.observation_dict = {c: X[y == c] for c in classes}
        self.maj_class_ = self.unique_classes_[0]

        n_max = max(sizes)
        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, max(sizes)))

    def fit_sample(self, X, y):

        self.fit(X, y)
      
        for i in range(1, len(self.observation_dict)):
            current_class = self.unique_classes_[i]
            self.n = len(self.observation_dict[0]) - len(self.observation_dict[current_class])
           
            reshape_points, reshape_labels = self.reshape_observations_dict()
            oversampled_points, oversampled_labels = self.generate_samples(reshape_points, reshape_labels, current_class)
            self.observation_dict = {cls: oversampled_points[oversampled_labels == cls] for cls in self.unique_classes_}
 
        reshape_points, reshape_labels = self.reshape_observations_dict()

        return reshape_points, reshape_labels
    def generate_samples(self, X, y, minority_class=None):
  
        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()
        self.compute_weight(majority_points,minority_points)
        min_to_min_distances = distance_matrix(minority_points, minority_points, self.p_norm)
        appended = []
        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            min_to_min_dis = min_to_min_distances[i]
            asc_min_index = np.argsort(min_to_min_dis)

            if self.k_min <= len(minority_points)-1:
                knn_index = asc_min_index[1:self.k_min + 1]
            else:
                knn_index = asc_min_index[1:]
           
            for _ in range(int(self.gi[i])):
               
                random_tow = knn_index[np.random.choice(range(len(knn_index)), size=2,replace=False)]
               
                vector1 = minority_point + (2 * np.random.rand() - 1)*(minority_points[random_tow[0]]-minority_point)
                vector2 = minority_point + (2 * np.random.rand() - 1) * (minority_points[random_tow[1]] - minority_point)
    
                direction_vector = vector1+ vector2
                direction_unit_vector = direction_vector / norm(direction_vector)
                new_data = minority_point + direction_unit_vector * np.random.rand() * self.radii[i]
          
                appended.append(new_data)

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])
        return points,labels


    def compute_weight(self,majority_points,minority_points):

        self.n = len(majority_points) - len(minority_points)
        min_to_maj_distances = distance_matrix(minority_points, majority_points, self.p_norm)
        min_to_min_distances = distance_matrix(minority_points, minority_points, self.p_norm)

        self.radii = np.zeros(len(minority_points))
        for i in range(len(minority_points)):
           
            asc_index = np.argsort(min_to_maj_distances[i])
            radius = min_to_maj_distances[i, asc_index[0]]
            self.radii[i] = radius

        min_weight = np.zeros(len(minority_points))

        regoin_count = np.zeros(len(minority_points))

        min_to_kmaj_dis = np.zeros(len(minority_points))

        for i in range(len(minority_points)):
            for j in range(len(min_to_min_distances[i])):
                if min_to_min_distances[i][j] < self.radii[i] and i != j:
                    regoin_count[i] = regoin_count[i] + 1

    
            asc_max_index = np.argsort(min_to_maj_distances[i])
            dis = min_to_maj_distances[i, asc_max_index[:self.k_maj - 1]]
            min_to_kmaj_dis[i] = np.mean(dis)
            min_weight[i] = 1.0  / (min_to_kmaj_dis[i] * (regoin_count[i] + 1.0) / self.radii[i])

        max_weight = np.percentile(min_weight, q=self.max_weight_percentile)
        for i in range(len(minority_points)):
            if min_weight[i] >= max_weight:
                min_weight[i] = max_weight

        weight_sum = np.sum(min_weight)
        for i in range(len(minority_points)):
            min_weight[i] = min_weight[i] / weight_sum
      
        self.gi = np.rint(min_weight * self.n).astype(np.int32)



    def reshape_observations_dict(self):
    
        reshape_points = []
        reshape_labels = []

        for cls in self.observation_dict.keys():
            if len(self.observation_dict[cls]) > 0:
                
                reshape_points.append(self.observation_dict[cls])
               
                reshape_labels.append(np.tile([cls], len(self.observation_dict[cls])))

        reshape_points = np.concatenate(reshape_points)
        reshape_labels = np.concatenate(reshape_labels)

        return reshape_points, reshape_labels
