# -*- coding: utf-8 -*-
__project__ = 'jdiscovery'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '16 apr 2020'
__comment__ = '''Classes for the creation of Training and Test sets according to different formats'''
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import numpy as np
from tqdm import tqdm
from typing import Dict
import json
import re
import string

STOPWORDS = set(stopwords.words('english'))


class DocumentMissing(Exception):

    def __init__(self, *args):
        if args:
            self.document = args[0]
        else:
            self.document = None

    def __str__(self):
        if self.document:
            return "DocumentMissing: Document `{}` is not available in the corpus".format(self.document)
        else:
            return "DocumentMissing"


class TrainingCorpus(object):

    def __init__(self):
        super().__init__()
        self.labels = []
        self.target = {}
        self.noun_chunks = defaultdict(lambda: 0)
        self.document_chunks = {}
        self._label_index = {}

    def load(self, corpus_file_path: str):
        with open(corpus_file_path, 'r') as in_file:
            data = json.load(in_file)
        self.docs = data['docs']
        self.texts = data['texts']
        self.tokens = data['tokens']
        self.labels = data['labels']
        self.target = dict([(int(x), y) for x, y in data['target'].items()])
        self._index = dict([(x, i) for i, x in enumerate(self.docs)])
        self._label_index = dict([(c, i) for i, c in enumerate(self.labels)])
        self.token_count = self.__token_to_count()
    
    def __token_to_count(self) -> Dict[str, int]:
        tokens_list = [token for doc_tokens in self.tokens for token in doc_tokens]
        token_to_count_map = dict(Counter(tokens_list))
        return token_to_count_map
    
    @property
    def size(self) -> int:
        return len(self.docs)
    
    @staticmethod
    def _check(token):
        if token in string.punctuation:
            return False
        elif token.startswith("'"):
            return False
        elif token == '--':
            return False
        elif token == '``':
            return False
        elif token == '“':
            return False
        elif token == '”':
            return False
        elif token in STOPWORDS:
            return False
        else:
            return True

    @staticmethod
    def tokenize(text):
        t = re.sub(r'(?<=\S)\.(?=\w)', '. ', text)
        return [x for x in word_tokenize(t) if TrainingCorpus._check(x)]

    def get_text(self, document_id: int) -> str:
        try:
            return self.texts[self._index[document_id]]
        except KeyError:
            raise DocumentMissing(document_id)

    def get_tokens(self, document_id: int) -> list:
        try:
            return self.tokens[self._index[document_id]]
        except KeyError:
            raise DocumentMissing(document_id)

    def detect_chunks(self, spacy_model: str = 'en_core_web_sm'):
        """
        Find noun chunks in document texts. Requires Spacy (https://spacy.io).
        Feed two internal indexes: noun_chunks providing the frequency of each chunk
        in the whole corpus; document_chunks: gives the list of chunks per document
        :param: name of the spacy model to use
        :return: None
        """
        nlp = spacy.load(spacy_model)
        data = tqdm(list(enumerate(self.texts)))
        for i, text in data:
            chunks = []
            for chunk in nlp(text).noun_chunks:
                c = "_".join(self.tokenize(chunk.text))
                chunks.append(c)
                self.noun_chunks[c] += 1
            self.document_chunks[self.docs[i]] = chunks

    @staticmethod
    def find_mix(seq, subseq):
        n = len(seq)
        m = len(subseq)
        for i in range(n - m + 1):
            if seq[i] == subseq[0] and seq[i:i + m] == subseq:
                yield range(i, i + m)

    def save_chunks(self, chunk_file: str):
        with open(chunk_file, 'w') as out:
            data = {'chunks': dict(self.noun_chunks), 'doc_chunks': self.document_chunks}
            json.dump(data, out)

    def load_chunks(self, chunk_file: str):
        with open(chunk_file, 'r') as in_file:
            data = json.load(in_file)
        self.noun_chunks = data['chunks']
        for k, v in data['doc_chunks'].items():
            self.document_chunks[int(k)] = v

    def get_chunk_document(self, document_id: int, threshold: int = 10) -> list:
        """
        Returns a version of the document build from tokens where chunks with frequency
        higher or equal than threshold are taken as single words
        :param document_id: the id (not position) of the document
        :param threshold: minimum frequency for noun chunks
        :return: list of tokens where noun chunks are taken as single words
        """
        try:
            chunks = [x for x in self.document_chunks[document_id] if self.noun_chunks[x] >= threshold]
            tokens = self.tokens[self._index[document_id]]
            if len(chunks) > 0:
                for k_chunk in chunks:
                    chunk = k_chunk.split('_')
                    replacements = [r for r in TrainingCorpus.find_mix(tokens, chunk)]
                    l, f = 0, []
                    while l < len(tokens):
                        replaced = False
                        for r in replacements:
                            if l in r:
                                replaced = True
                                f.append(chunk)
                                l += len(chunk)
                                break
                            else:
                                pass
                        if not replaced:
                            f.append(tokens[l])
                            l += 1
                    new_tokens = []
                    for x in f:
                        if isinstance(x, list):
                            new_tokens.append("_".join(x))
                        else:
                            new_tokens.append(x)
                    tokens = new_tokens
                return tokens
            else:
                return tokens
        except KeyError:
            raise DocumentMissing(document_id)

    def get_one_hot_target(self) -> np.ndarray:
        y = np.zeros((self.size, len(self.labels)))
        for doc, annotations in self.target.items():
            for a in annotations:
                y[self._index[doc], self._label_index[a]] = 1
        return y

    def get_train_test_data(self, test_size: float = 0.2, random_state=3) -> tuple:
        y = self.get_one_hot_target()
        X_train, X_test, y_train, y_test = train_test_split(self.docs, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test





