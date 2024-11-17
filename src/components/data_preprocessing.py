import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        try:
            # Handle None or non-string values
            if text is None or not isinstance(text, str):
                return ""
                
            # Clean the text
            text = str(text)  # Convert to string if not already
            text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces
            text = text.lower()  # Convert to lowercase
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Tokenize and remove stop words
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            
            # Return cleaned text
            return ' '.join(tokens) if tokens else ""
            
        except Exception as e:
            logging.error(f"Error in clean_text: {str(e)}")
            raise CustomException(e, sys)