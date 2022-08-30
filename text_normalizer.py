import re
import nltk
import spacy
import unidecode

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
porter = PorterStemmer()

def remove_html_tags(text):
    text_html_rm = BeautifulSoup(text, 'html.parser')
    return text_html_rm.get_text()

def stem_text(text):
    token_words = tokenizer.tokenize(text, return_str=False)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    return " ".join(stem_sentence)

def lemmatize_text(text):
    test_list=[]
    for token in nlp(text):
        test_list.append(token.lemma_)
    return " ".join(test_list)


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP, lower_case=False):
    # Create a regular expression from the dictionary keys
    # REs are separated with |.
    regex = re.compile("|".join(map(re.escape, contraction_mapping.keys())))
    # For each match, look-up corresponding value in dictionary.
    # Each REs between | is scanned until one matched. 
    if lower_case:
        return regex.sub(lambda mo: contraction_mapping[mo.string[mo.start():mo.end()]], text.lower())
    else:
        return regex.sub(lambda mo: contraction_mapping[mo.string[mo.start():mo.end()]], text)


def remove_accented_chars(text):
    return unidecode.unidecode(text)


def remove_special_chars(text, remove_digits=True):
    if remove_digits:
        return re.sub("[^a-zA-Z\s]", "", text)
    else:
        return re.sub("[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    return ' '.join(filtered_tokens)


def remove_extra_new_lines(text):
    return ' '.join(text.split())


def remove_extra_whitespace(text):
    return ' '.join(text.split())
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc, lower_case=True)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus