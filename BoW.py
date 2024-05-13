import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Baixando os recursos do NLTK necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Carregando o arquivo CSV
data = pd.read_csv('C:\\Users\\finbank_data.csv')

# Subconjunto com os primeiros 50 registros
subset = data.head(50)

# Função para conseguirmos realizar o pré-processamento do texto
def preprocess_text(text):
    sentences = sent_tokenize(text)  # Segmentação por sentenças
    words = word_tokenize(text)      # Segmentação por unidades (Tokenização)
    
    # Capitalização e remoção de stopwords
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
    
    # Lematização [descomentar aqui e comentar o stemmer caso queiramos o lemmatizer]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Stemming [descomentar aqui e comentar o lemmatizer caso queiramos o stemmer]
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# Aplicando o pré-processamento aos primeiros 50 registros
subset['Processed_Text'] = subset['text'].apply(preprocess_text)

# Criando a bag of words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(subset['Processed_Text'])

# Criando um DataFrame para armazenar a matriz do BoW
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Exibindo a matriz do BoW
print(bow_df)
