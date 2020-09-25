import numpy as np
import pandas as pd
from typing import List, Union, Dict
from lime.explanation import Explanation
from lime.lime_text import LimeTextExplainer
from training import TrainingCorpus
from model import Model


class LimeTermFinder(object):

    def __init__(self,
                 model: Model,
                 data: TrainingCorpus,
                 min_fts: int = 30,
                 max_fts: int = 100,
                 max_fts_alpha: float = 0.5) -> None:
        """
        Initialize `LimeTermFinder` object

        Parameters
        ----------
        model: Model
            `Model` instance
        data: : TrainingCorpus
            `TrainingCorpus` instance
        min_fts: int
            Constraint on the number of features to account
            when explaining an text instance using LIME. If a text 
            `t` has a number of features (singleton words and noun chunks)
            less or equal to `min_fts` we don't filter out any feature
            before explaining `t`
        max_fts: int
            Upper bound on the maximum number of features to account
            when explaining a text instance `f` using LIME. We truncate the 
            number of features (singleton words and noun chunks) of `t` so that the
            maximum number of features taken into account in LIME is at most
            `max_fts`
        max_fts_alpha: float
            Minimum percentage of features to account for LIME. See 
            `self.__select_features` and
            `self.__compute_no_of_features` for more details.
        """
        self.model = model
        self.data = data
        self.min_fts = min_fts
        self.max_fts = max_fts
        self.max_fts_alpha = max_fts_alpha
        self.explainer = LimeTextExplainer(
            feature_selection='none', split_expression=' ')

    def get_explanation(self,
                        doc_idx: int,
                        label: int,
                        num_samples: int = 5000) -> Explanation:
        """
        Get an explanation for an instance using `LIME`

        Parameters
        ----------
        doc_idx: int
            Index (not id) of a document to be explained
        label: int
            Label for which we explain the input document
        num_samples: int
            Number of perturbed samples generated in order to explain the
            input document

        Returns
        -------
        result: Explanation
            `lime.explanation.Explanation` object
        """
        # retrieve document from idx
        doc_id = self.data.docs[doc_idx]
        original_text_tokens = self.data.get_chunk_document(
            doc_id, threshold=0)

        # select features
        no_of_features = self.__compute_no_of_features(original_text_tokens)
        selected_features = self.__select_features(doc_id, no_of_features)
        selected_text = ' '.join(
            [token for token in original_text_tokens if token in selected_features])

        # for reproducibility
        np.random.seed(len(original_text_tokens))

        # get explanation
        exp = self.explainer.explain_instance(selected_text,
                                              lambda x: self.__predict_fn_wrapper(label,
                                                                                  x,
                                                                                  selected_features,
                                                                                  original_text_tokens),
                                              num_samples=num_samples,
                                              labels=[1])

        return exp

    def get_relevant_terms(self,
                           doc_idx: int,
                           label: int,
                           num_samples: int = 5000,
                           threshold: float = 0.9) -> Dict[str, float]:
        """
        Retrieve the relevant terms for an instance according to `LIME`

        Parameters
        ----------
        doc_idx: int
            Index (not id) of document to be explained
        label: int
            Label for which we explain the input document
        num_samples: int
            Number of perturbed samples generated in order to explain the
            input document
        threshold: float
            Threshold used to filter out features (singleton words or compound tokens)
            whose contribution to the predicted label is marginal

        Returns
        -------
        result: Dict[str, float]
            Dictionary where keys are relevant features (singleton words or compound tokens)
            and values are their weights according to `LIME`
        """
        # get explanation
        exp = self.get_explanation(doc_idx, label, num_samples)
        exp_list = exp.as_list(label=1)

        # build a pandas series out of exp
        exp_series = pd.DataFrame(exp_list,
                                  columns=['feature', 'weight']
                                  ).set_index('feature')['weight']

        # get relevant terms
        relevant_terms = self.__relevant_terms_by_threshold(exp_series,
                                                            threshold=threshold)

        return relevant_terms

    def __compute_no_of_features(self,
                                 text_tokens: List[str]) -> int:
        """
        Given list of tokens representing a text, returns the number
        of features to be explained by `LIME` via linear interpolation

        Parameters
        ----------
        text_tokens: List[str]
            Tokenized text (including compound tokens) 

        Returns
        -------
        result: int
            Number of features to be explained
            using `LIME`
        """
        text_feature_count = len(set(text_tokens))
        alpha = np.interp(text_feature_count,
                          [self.min_fts, self.max_fts*(1/self.max_fts_alpha)],
                          [1, self.max_fts_alpha])

        no_of_features = int(alpha * text_feature_count)
        no_of_features = min(no_of_features, self.max_fts)

        return no_of_features

    def __select_features(self,
                          doc_id: int,
                          no_of_features: int) -> List[str]:
        """
        Select which features will be explained using `LIME`.
        We sort tokens (singleton words and compound words)by their 
        frequency. After that, we pick features
        prioritising compound tokens w.r.t to singleton words

        Parameters
        ----------
        doc_id: int
            Index (not id) of input document
        no_of_features: int
            Desired number of features to be picked from 
            `doc_id`

        Returns
        -------
        result: List[str]
            List of selected features
        """
        text_tokens = self.data.get_chunk_document(doc_id, threshold=0)

        selected_features = []
        text_feature_count = len(set(text_tokens))

        # in the first case we take all features in text_tokens
        if text_feature_count == no_of_features:
            selected_features = text_tokens

        else:
            # for each chunk and word in text_tokens
            # retrieve their corresponding frequency
            doc_chunks = self.data.document_chunks[doc_id]
            chunk_to_count = {}
            word_to_count = {}

            for token in text_tokens:
                if token in doc_chunks:
                    chunk_to_count[token] = self.data.noun_chunks[token]
                else:
                    word_to_count[token] = self.data.token_count[token]

            # take features starting from chunks
            # (and then possibly from words) based on
            # their frequency
            if chunk_to_count:
                chunk_to_count = chunk_to_count.items()
                sorted_chunk_to_count = sorted(
                    chunk_to_count, key=lambda x: x[1], reverse=True)
                selected_features = [chunk for chunk,
                                     _ in sorted_chunk_to_count[:no_of_features]]
                selected_features_count = len(selected_features)

                # whether the number of selected chunks is
                # not enough we take the remaining
                # feature from the word_to_count dict (based on freq)
                if selected_features_count < no_of_features:
                    remaining_ft_count = no_of_features - selected_features_count
                    word_to_count = word_to_count.items()
                    sorted_word_to_count = sorted(
                        word_to_count, key=lambda x: x[1], reverse=True)
                    selected_features += [word for word,
                                          _ in sorted_word_to_count[:remaining_ft_count]]
            else:
                word_to_count = word_to_count.items()
                sorted_word_to_count = sorted(
                    word_to_count, key=lambda x: x[1], reverse=True)
                selected_features = [word for word,
                                     _ in sorted_word_to_count[:no_of_features]]

        return selected_features

    def __relevant_terms_by_threshold(self,
                                      weight_series: pd.Series,
                                      threshold: float = 0.9) -> Dict[str, float]:
        """
        Filter out features whose contribution to the
        predicted label is marginal according to `LIME`

        Parameters
        ----------
        weight_series: pd.Series
            pandas Series where indexes are features/tokens
            (singleton words or compound tokens) and values are
            weights
        threshold: float
            Threshold used to filter out features with marginal
            weights.

        Returns
        -------
        result: Dict[str, float]
            Dictionary where keys are relevant features/tokens (singleton words
            or compound words) and values are their weights according to `LIME`
        """
        filtered_series = weight_series[weight_series > 0]
        sorted_series = filtered_series.sort_values(ascending=False)

        if threshold == 1:
            # return all positive weights
            return sorted(sorted_series.index)
        else:
            total = sorted_series.sum()

            if total == 0:
                # no positive weights
                return []
            else:
                curr_idx = 0
                curr_sum = 0

                while curr_sum/total < threshold:
                    curr_sum += sorted_series.iloc[curr_idx]
                    curr_idx += 1
                feature_weight_series = sorted_series.iloc[0:curr_idx]
                feature_weight_dict = feature_weight_series.to_dict()
                return feature_weight_dict

    def __map_text(self,
                   text: str,
                   selected_features: List[str],
                   original_text_tokens: List[str]) -> str:
        """
        Given a tokenized document `original_text_tokens`, a set of
        selected features `selected_features` (`selected_features`
        is a subset of `set(original_text_tokens)`) and a pertubed text
        `text` (tokens in `text` are a subset of those in
        `selected_features`) generated using LIME, return a new
        set of tokens from `original_text_tokens` where tokens dropped
        from `selected_features` in order to get `text`
        are also dropped in `original_text_tokens`.

        Parameters
        ----------
        text: str
            Input string representing a pertubed text
            generated during a LIME explanation
        selected_features: List[str]
            List of tokens for which we want an explanation
        original_text_tokens: List[str]
            Tokenized version of the initial document

        Returns
        -------
        result: str
            A list of tokens from `original_text_tokens` where tokens dropped
            in `selected_features` in order to get `text`
            are also dropped in `original_text_tokens`.
        """
        tokenized_text = text.split()
        dropped_tokens = [
            token for token in selected_features if token not in tokenized_text]
        new_tokenized_text = [
            token for token in original_text_tokens if token not in dropped_tokens]
        new_text = ' '.join(new_tokenized_text)

        return new_text

    def __predict_fn_wrapper(self,
                             label: int,
                             texts: Union[str, List[str]],
                             selected_features: List[str],
                             original_text_tokens: List[str]) -> np.ndarray:
        """
        Given an output label index `label`, a tokenized document `original_text_tokens`, 
        a set of selected features `selected_features` (`selected_features`
        is a subset `set(original_text_tokens)`) and a pertubed text
        `text` (tokens in `text` are a subset of those in
        `selected_features`) generated using LIME, firstly, it returns a new
        set of tokens from `original_text_tokens` where tokens dropped
        from `selected_features` in order to get `text`
        are also dropped in `original_text_tokens`. The resulting text is then
        fed to the model and  2-dimensional output vector
        containing the predicted probability for `label`
        (i.e `[1 - proba(label), proba(label)]`) is then returned.

        Parameters
        ----------
        label: int
            Output label index
        text: str
            Input string representing a pertubed text
            generated during a LIME explanation
        selected_features: List[str]
            List of tokens for which we want an explanation
        original_text_tokens: List[str]
            Tokenized version of the initial document

        Returns
        -------
        result: np.ndarray
            2-dim vector containing the predicted probability for `label`
        """
        if isinstance(texts, str):
            texts = [texts]

        new_texts = list(map(lambda x: self.__map_text(
            x, selected_features, original_text_tokens), texts))
        pred_proba = self.model.predict_fn(new_texts)
        label_pred_proba = pred_proba[:, label].reshape(-1, 1)
        label_pred_proba = np.hstack(
            ((1 - label_pred_proba), label_pred_proba))
        return label_pred_proba
