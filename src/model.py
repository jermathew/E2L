import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from abc import ABC, abstractmethod
from collections import defaultdict
from training import TrainingCorpus
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

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


class TensorflowTfIdfModel(Model):

    def __init__(self, 
                 model_filepath: str, 
                 vectorizer_filepath: str):
        
        model = load_model(model_filepath)
        self.input_len = model.input.shape[1]
        with open(vectorizer_filepath, 'rb') as fp:
            self.vectorizer = dill.load(fp)
        super().__init__(model)


    def label_to_data_idx(self,
                          data: TrainingCorpus) -> Dict[int, List[int]]:

        label_to_data_idx_map = defaultdict(list)

        docs_tokens = [data.get_tokens(doc_id) for doc_id in data.docs]
        input_data = np.array([self.__vectorize_text(
            doc_tokens) for doc_tokens in docs_tokens])
        predictions = self.model.predict(input_data)
        # round to the nearest integer
        predictions = np.rint(predictions)
        nonzeros = predictions.nonzero()

        for data_idx, label_idx in zip(*nonzeros):
            label_to_data_idx_map[label_idx].append(data_idx)

        return label_to_data_idx_map
    

    def predict_fn(self,
                   texts: List[str]) -> np.ndarray:

        input_data = np.array([self.__vectorize_text(text.split()) for text in texts])
        predict_matrix = self.model.predict(input_data)
        return predict_matrix
    
    
    def __vectorize_text(self, 
                         words: List[str])-> np.ndarray:
        
        tokens = []
        
        # split words into single terms
        for term in words:
            if '_' in term:
                # term is a noun chunk
                tokens += term.split('_')
            else:
                # term is a single word/term
                tokens.append(term)
        
        text_vector = self.vectorizer.transform([' '.join(tokens)]).toarray().reshape(-1)
        return text_vector

    
class BertModel(Model):

    def __init__(self, 
                 dir_path: str, 
                 batch_size: int = 256,
                 use_cuda: bool = False, from_tf=True):
        
        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        model = AutoModelForSequenceClassification.from_pretrained(dir_path, from_tf=from_tf)
        if self.use_cuda:
            model.cuda()
        super().__init__(model)
    
    
    def __chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

   
    def label_to_data_idx(self,
                          data: TrainingCorpus) -> Dict[int, List[int]]:
            
        label_to_data_idx_map = defaultdict(list)
        predictions_list = []
        texts = [' '.join(data.get_tokens(doc_id)) for doc_id in data.docs]
        
        for batch in self.__chunks(texts, self.batch_size):
            encoded_batch = self.tokenizer(batch,
                                           padding=True, 
                                           truncation=True, 
                                           return_tensors='pt')
            if self.use_cuda:
                encoded_batch.to('cuda')
            
            predictions_batch = self.model(**encoded_batch).logits
            predictions_batch = F.softmax(predictions_batch, dim=-1)
            
            if self.use_cuda:
                predictions_batch = predictions_batch.cpu()
            
            predictions_batch = predictions_batch.detach().numpy()
            predictions_list.append(predictions_batch)
        
        # concatenate all batches
        predictions = np.concatenate(predictions_list, axis=0)
        assert predictions.shape[0] == len(texts)
        
        # round to the nearest integer
        predictions = np.rint(predictions)
        nonzeros = predictions.nonzero()

        for data_idx, label_idx in zip(*nonzeros):
            label_to_data_idx_map[label_idx].append(data_idx)

        return label_to_data_idx_map
    

    def predict_fn(self,
                   texts: List[str]) -> np.ndarray:
        
        if self.use_cuda:
            # empty cache
            torch.cuda.empty_cache()
            
        preprocessed_texts = [self.__preprocess_text(t) for t in texts]
        pred_list = []
        
        for batch in self.__chunks(preprocessed_texts, self.batch_size):
            encoded_batch = self.tokenizer(batch, 
                                           padding=True, 
                                           truncation=True,
                                           return_tensors='pt')
            if self.use_cuda:
                encoded_batch.to('cuda')
            
            pred_batch = self.model(**encoded_batch).logits
            pred_batch = F.softmax(pred_batch, dim=-1)
            
            if self.use_cuda:
                pred_batch = pred_batch.cpu()
            
            pred_batch = pred_batch.detach().numpy()
            pred_list.append(pred_batch)
        
        # concatenate all batches
        pred = np.concatenate(pred_list, axis=0)
        assert pred.shape[0] == len(preprocessed_texts)
        
        return pred
    
    
    def __preprocess_text(self,
                          text: str)-> str:
        tokens = []
        
        # split words into single terms
        for term in text.split():
            if '_' in term:
                # term is a noun chunk
                tokens += term.split('_')
            else:
                # term is a single word/term
                tokens.append(term)
        
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text