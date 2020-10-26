"""utils for text processing and language checking
"""
import re
import string

from tempfile import TemporaryDirectory

import fitz
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textract
import easyocr
from langdetect import detect

from pdf2image import convert_from_path


def extract_text(file_path):
    """extract the text from a given text document or image

    Parameters
    ----------
    file_path : str
        string to the current text file

    Returns
    -------
    str
        the extracted string/text from document
    """

    try:
        text = ' '
        with fitz.open(file_path) as doc:
            for page in doc:
                text+= page.getText()
    
        if text == ' ':
            print('_______________USING_TEXTRACT_______________')
            text = textract.process(file_path, method='tesseract', encoding='utf-8')
            text = text.decode('utf8')
        
    except:
        reader = easyocr.Reader(['de', 'en', 'es', 'it'], gpu=True, model_storage_directory='text_classifier/models/ocr_model', download_enabled=True)
        doc_text = list()

        print('_______________USING_easyOCR_______________')
        with TemporaryDirectory() as path:
            images_from_path = convert_from_path(file_path, fmt='png', output_folder=path, paths_only=True, dpi=300, grayscale=True)
            for image in images_from_path:
                # min_size (int, default = 10) - Filter text box smaller than minimum value in pixel
                results = reader.readtext(image, min_size=40) # decoder='beamsearch', min_size=50, beamWidth=2
                # results = sorted(results, key=lambda x: x[0][0])  # sort text from left to right
                for (_, text, _) in results:
                    text = ''.join(text)
                    doc_text.append(text)

        text = ' '.join(doc_text)
        
    return text

def check_language(text):
    """checks the language of the given string/text

    Parameters
    ----------
    text : str
        the input string/text

    Returns
    -------
    str
        the language of the string/text (de, en)
    """
    if detect(text) == 'de':
        language = 'de'
    elif detect(text) == 'en':
        language = 'en'
    else:
        language = None
    return language

def clean_str(text):
    """clean the string from unusable characters, numbers, etc.

    Parameters
    ----------
    text : str
        the input string/text

    Returns
    -------
    str
        the cleaned string/text
    """
    text = str(text)
    # replace word binding
    # replace Umlaute
    text = re.sub(r'ẞ', 'ss', text)
    text = text.lower()
    text = re.sub(r'sz', 'ss', text)
    text = re.sub(r'ä', 'ae', text)
    text = re.sub(r'ö', 'oe', text)
    text = re.sub(r'ü', 'ue', text)
    # URLs
    text = re.sub(r'url?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'html', '', text, flags=re.MULTILINE)
    text = re.sub(r'url', '', text, flags=re.MULTILINE)
    # Linebreaks
    text = re.sub(r'(?<!\\)(\\\\)*\\n|\n', '', text, flags=re.MULTILINE)
    text = text.replace('- ', '').replace(' -', '').replace('-', '')
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.MULTILINE)
    # multi letters >2 times
    text = re.sub(r'(.)\\1{2,}', '', text, flags=re.MULTILINE)
    # alphabetic 
    text =  re.sub(r"\b[a-zA-Z]\b", '', text, flags=re.MULTILINE)
    # remove any whitespaces
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE)
    return text

def preprocess_texts(text, language, filter_stop_words):
    """remove stopwords and punctuations

    Parameters
    ----------
    text : str
        the input string/text
    language : str
        the language of the current string/text (de, en)
    filter_stop_words : bool, optional
        removes stopwords from a given dictonary, by default True

    Returns
    -------
    str
        the filtered string/text
    """
    if language == 'de':
        stopWords = stopwords.words('german')
    elif language == 'en':
        stopWords = stopwords.words('english')

    text = clean_str(text)

    word_tokens = word_tokenize(text)

    if filter_stop_words:
        filtered_tokens = [word for word in word_tokens if not word in stopWords and not word in string.punctuation]
    else:
        filtered_tokens = [word for word in word_tokens if not word in string.punctuation]
    concatenated_sentences = ' '.join(filtered_tokens)
    text = concatenated_sentences
    return text

