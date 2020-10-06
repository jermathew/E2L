import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from training import TrainingCorpus
from typing import List, Tuple, Dict

__pdoc__ = {}
__pdoc__[
    'Sampler.__init__'] = 'Abstract class for sampling data instances (documents)'
__pdoc__[
    'KMeansSampler.__init__'] = 'Class for sampling data instances (documents) based on KMeans'


class Sampler(ABC):

    def __init__(self,
                 data: TrainingCorpus,
                 label_to_data_idx: Dict[int, List[int]]) -> None:
        """
        Initialize `Sampler` object

        Parameters
        ----------
        data: TrainingCorpus
            `TrainingCorpus` instance
        label_to_data_idx: Dict[int, List[int]]
            Dictionary where keys are label indexes `l` and values are index
            of documents (not doc ids) whose predicted labels correspond to `l`
        """
        self.data = data
        self.label_to_data_idx = label_to_data_idx
        super().__init__()

    @abstractmethod
    def sample_data(self,
                    label: int,
                    size: int,
                    random_state: int) -> List[int]:
        """
        Sample documents by predicted label

        Parameters
        ----------
        label: int
            label index from which we take the list of
            document which we want to sample from
        size: int
            Sample size
        random_state: int
            Random state (for reproducibility)

        Returns
        -------
        result: List[int]
            Indexes (not ids) of a sample of documents
        """
        pass


class KMeansSampler(Sampler):

    def __init__(self,
                 data: TrainingCorpus,
                 embeddings: np.ndarray,
                 label_to_data_idx: Dict[int, List[int]],
                 min_size: int = 1000,
                 max_size: int = 5000,
                 max_size_alpha: float = 0.5) -> None:
        """
        Initialize `KmeansSampler` object

        Parameters
        ----------
        data: `TrainingCorpus`
            `TrainingCorpus` instance
        label_to_data_idx: Dict[int, List[int]]
            Dictionary where keys are label indexes `l` and values are index
            of documents (not doc ids) whose predicted labels correspond to `l`
        min_size: int
            Constraint on the input documents `D` size. If `size(D)` is 
            below or equal to `min_size` we return `D` (in other words there is no sampling).
        max_size: int
            Upper bound to the size of the resulting sample of data. The resulting 
            sample of data must contains at most `max_size` instances
        max_size_alpha: float
            Float value in (0,1) representing the minimum proportion of documents 
            to be picked from `D`. See `self.__get_sample_size` for more details.
        """
        self.embeddings = embeddings
        self.min_size = min_size
        self.max_size = max_size
        self.max_size_alpha = max_size_alpha
        super().__init__(data, label_to_data_idx)

    def sample_data(self,
                    label: int,
                    size: int = None,
                    random_state: int = 3) -> List[int]:

        data_idxs = self.label_to_data_idx[label]

        # retrieve embeddings
        data_emb = self.embeddings[data_idxs, :]

        # check if data_emb contains duplicate embeddings
        data_idxs, data_emb = self.__remove_duplicates(data_idxs, data_emb)

        data_size = len(data_idxs)

        # compute sample size
        if size is None:
            size = self.__get_sample_size(data_size)
        else:
            if size > 0:
                size = min(size, data_size)
            else:
                raise ValueError("size must be greater than zero")

        if size == data_size:
            return data_idxs
        else:
            # cluster data
            kmeans = KMeans(n_clusters=size,
                            random_state=random_state, n_jobs=2)
            kmeans.fit(data_emb)

            # retrieve cluster labels and centroids
            cluster_labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            # build a DataFrame out of embeddings and cluster labels
            df = pd.DataFrame(data_emb, index=data_idxs)
            df['cluster_label'] = cluster_labels

            # group by cluster label
            grouped_df = df.groupby('cluster_label')

            # take a sample of data based on centroids
            sample_data_idxs = []
            for cluster_label, centroid in enumerate(centroids):
                cluster_data = grouped_df.get_group(
                    cluster_label).drop(columns='cluster_label')
                row_idx = self.__most_similar_row_idx(centroid, cluster_data)
                sample_data_idxs.append(row_idx)

            return sample_data_idxs

    def __remove_duplicates(self,
                            data_idxs: List[int],
                            data_emb: np.ndarray) -> Tuple[List[int], np.array]:
        """
        Remove duplicate data

        Parameters
        ----------
        data_idxs: List[int]
            List of document indexes (not ids)
        data_emb: np.ndarray
            Numpy ndarray containing `data_idxs`
            embeddings

        Returns
        -------
        result: Tuple[np.array, List[int]]
            First, indexes of documents whose embeddings are unique
            Second, embedding matrix without duplicate values
        """
        # check if data_emb contains duplicate embeddings
        data_emb_unique, unique_idxs, counts = np.unique(data_emb,
                                                         axis=0,
                                                         return_index=True,
                                                         return_counts=True)
        unique_idxs = unique_idxs.tolist()

        contains_duplicate = np.any(counts > 1)
        if contains_duplicate:
            data_emb = data_emb_unique
            data_idxs = [data_idxs[idx] for idx in unique_idxs]

        return (data_idxs, data_emb)

    def __most_similar_row_idx(self,
                               centroid: np.ndarray,
                               data_df: pd.DataFrame) -> int:
        """
        Given a centroid of a cluster `x`, returns the most similar
        instance among the points of cluster `x` based on l2 norm

        Parameters
        ----------
        centroid: List[int]
            Numpy vector representing a centroid
        data_df: pd.DataFrame
            DataFrame containing instances of cluster `x`

        Returns
        -------
        result: int
            Positional index of `data_df` correspoding to the most similar point
        """
        data_np = data_df.values
        row_idx_tmp = np.argmin(np.linalg.norm(data_np - centroid, axis=1))
        row_idx = data_df.iloc[row_idx_tmp].name
        return row_idx

    def __get_sample_size(self,
                          data_size: int) -> int:
        """
        Compute sample size based on the number of
        input documents by linear interpolation

        Parameters
        ----------
        data_size: int
            Input documents size which we sample from

        Returns
        -------
        result: int
            Resulting sample size
        """
        alpha = np.interp(data_size,
                          [self.min_size, self.max_size *
                              (1/self.max_size_alpha)],
                          [1, self.max_size_alpha])

        sample_size = int(alpha * data_size)
        sample_size = min(sample_size, self.max_size)

        return sample_size
