"""
The scikit-learn tfidf tool removes stop words by default. The list of stop words is here:
https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/feature_extraction/_stop_words.py

"""
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pubmed_parser as pp


def walk(path='./sample'):
    for parent, _, file_lst in os.walk(path):
        for file_name in file_lst:
            if file_name.endswith('xml'):
                yield os.path.join(parent, file_name)


if __name__ == '__main__':
    corpus = []

    # Read text.
    for path in walk():
        doc = pp.parse_pubmed_xml(path)
        text = doc['abstract']
        corpus.append(text)

    ################################################
    # Example with n-grams for n in [1, 2, 3].
    ################################################
    print('\n\n\nExample with n-grams for n in [1, 2, 3].')
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(corpus)
    ngrams = vectorizer.get_feature_names()

    print('# of n-grams:')
    print(collections.Counter([len(x.split()) for x in ngrams]))
    # Counter({3: 618, 2: 550, 1: 295})

    ################################################
    # Example with custom tokenizer.
    ################################################
    print('\n\n\nExample with custom tokenizer.')
    import spacy

    nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

    def spacy_tokenizer(doc):
        """
        Warning: The spacy tokenizer might convert words like "don't" into
        two words: "do", "n't".
        """
        doc = nlp(doc)
        # TODO: All filter should be done in one pass for speed.
        lst = [x for x in doc if x.is_alpha and not x.is_stop and not x.is_punct]
        # import ipdb; ipdb.set_trace()
        # Lemmatization.
        lst = [x.lemma_ for x in lst]
        # Remove anomolous words (includes space or is empty).
        lst = [x for x in lst if len(x) > 0 and len(x.split()) == 1]

        return lst

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), tokenizer=spacy_tokenizer)
    X = vectorizer.fit_transform(corpus)
    ngrams = vectorizer.get_feature_names()
    print('# of n-grams:')
    print(collections.Counter([len(x.split()) for x in ngrams]))
    # Counter({3: 370, 2: 334, 1: 214})

    num_docs = len(corpus)
    num_terms = len(ngrams)
    assert X.shape == (num_docs, num_terms)
    print('num-docs = {}, num-terms = {}'.format(num_docs, num_terms))

    # Find terms with highest avergage tfidf.
    weighted_X = np.asarray(X.mean(axis=0)).reshape(-1)
    assert weighted_X.shape == (num_terms,)
    index = np.argsort(weighted_X)[::-1] # Sort descending.
    terms_to_show = 20

    print('TOP TERMS')
    for i in range(terms_to_show):
        term_idx = index[i]
        term = ngrams[term_idx]
        avg_tfidf = weighted_X[term_idx]
        print('{:>10}\t{:>40}\t{:>10}'.format(i, term, avg_tfidf))
    print('')

    print('BOTTOM TERMS')
    for i in range(terms_to_show):
        i = -(i+1)
        term_idx = index[i]
        term = ngrams[term_idx]
        avg_tfidf = weighted_X[term_idx]
        print('{:>10}\t{:>40}\t{:>10}'.format(i, term, avg_tfidf))
    print('')

    np.random.seed(121)
    print('RANDOM TERMS')
    for i in sorted(np.random.choice(np.arange(num_terms), size=terms_to_show, replace=False)):
        term_idx = index[i]
        term = ngrams[term_idx]
        avg_tfidf = weighted_X[term_idx]
        print('{:>10}\t{:>40}\t{:>10}'.format(i, term, avg_tfidf))
    print('')

    print('min-avg-tfidf = {}, max-avg-tfidf = {}'.format(weighted_X.min(), weighted_X.max()))
