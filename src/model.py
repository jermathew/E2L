import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from abc import ABC, abstractmethod
from collections import defaultdict
from training import TrainingCorpus
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

__pdoc__ = {}
__pdoc__['Model.__init__'] = 'Abstract class for wrapping classification models'
__pdoc__['SklearnModel.__init__'] = 'Wrapper class for scikit-learn classification models based on embeddings'
__pdoc__['TensorflowModel.__init__'] = 'Wrapper class for keras-based tensorflow models based on embeddings'


class Model(ABC):

    def __init__(self,
                 model: Any) -> None:
        """
        Initialize `Model` object

        Parameters
        ----------
        model: Any
            Classification model
        """
        self.model = model
        super().__init__()

    @abstractmethod
    def label_to_data_idx(self,
                          data: TrainingCorpus) -> Dict[int, List[int]]:
        """
        Given a `TrainingCorpus` instance returns a dictionary which maps each
        label index `l` to a list of document indexes (not doc ids) for which
        the predicted label is `l`.

        Parameters
        ----------
        data: TrainingCorpus
            `TrainingCorpus` instance

        Returns
        -------
        result: Dict[int, List[int]]
            Dictionary where keys are label indexes `l` and values are a list
            of document indexes (not doc ids) for which the predicted label is `l`.
        """
        pass

    @abstractmethod
    def predict_fn(self,
                   texts: List[str]) -> np.ndarray:
        """
        Given a list of texts, returns their corresponding predicted
        probabilities

        Parameters
        ----------
        texts: List[str]
            List of texts

        Returns
        -------
        result: np.ndarray
            Numpy matrix `A=(n,m)` where `A[i,j]` is the predicted probability
            of concept `j` for text `i`.
        """
        pass


class TensorflowModel(Model):

    def __init__(self,
                 model_filepath: str,
                 word_idx_filepath: str) -> None:
        """
        Initialize TensorflowModel object

        Parameters
        ----------
        model: path to a keras model in h5 format
        word_idx_filepath: str
            Path to a csv file associating
            each word to an integer index
            (required for the embedding layer)
        """
        word_to_idx_df = pd.read_csv(word_idx_filepath, index_col='term')
        self.word_to_idx_map = word_to_idx_df.to_dict()['index']
        model = load_model(model_filepath)
        self.input_len = model.input.shape[1]
        super().__init__(model)

    def label_to_data_idx(self,
                          data: TrainingCorpus) -> Dict[int, List[int]]:

        label_to_data_idx_map = defaultdict(list)

        docs_tokens = [data.get_tokens(doc_id) for doc_id in data.docs]
        docs_sequences = np.array([self.__tokens_to_sequence(
            doc_tokens) for doc_tokens in docs_tokens])
        predictions = self.model.predict(docs_sequences)
        # round to the nearest integer
        predictions = np.rint(predictions)
        nonzeros = predictions.nonzero()

        for data_idx, label_idx in zip(*nonzeros):
            label_to_data_idx_map[label_idx].append(data_idx)

        return label_to_data_idx_map

    def predict_fn(self,
                   texts: List[str]) -> np.ndarray:

        texts_sequences = np.array(
            [self.__tokens_to_sequence(text.split()) for text in texts])
        predict_matrix = self.model.predict(texts_sequences)
        return predict_matrix

    def __tokens_to_sequence(self,
                             text_tokens: List[str]) -> np.ndarray:
        """
        Converts a list of strings into a list of
        integer sequences according to `self.word_to_idx_map`

        Parameters
        ----------
        text_tokens: List[str]
            List of strings

        Returns
        -------
        result: np.ndarray
            Matrix containing for each item in
            `text_tokens` a sequence of integer indexes
            according to `self.word_to_idx_map`
        """
        sequence = []

        for compound_word in text_tokens:
            if '_' in compound_word:
                # compound_word is a noun chunk
                tokens = compound_word.split('_')
            else:
                # compound_word is a single word/token
                tokens = [compound_word]

            for token in tokens:

                if token in self.word_to_idx_map:
                    sequence.append(self.word_to_idx_map[token])

        padded_sequence = pad_sequences(
            [sequence], maxlen=self.input_len).reshape(-1)

        return padded_sequence
