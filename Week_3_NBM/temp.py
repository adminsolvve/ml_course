import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)





def calc_tf(docum):
    """
    parameters: docum - List of words
    returns: Counter object with TF for all words in docum
    """
    tf = collections.Counter(docum)
    for i in tf:
        tf[i] = tf[i]/float(len(docum))
    return tf

def calc_idf(word, corpus):
    """
    parameters: corpus - List of texts
                word - or which to calculate IDF
    returns: value, idf for word in corpus            
    """
    word_in_doc_count = sum([1.0 for i in corpus if word in i])
    if word_in_doc_count > 0:
        idf = len(corpus)/word_in_doc_count
    else:
        idf = 0
    return math.log10(idf)

texts = [['pasta', 'la', 'vista', 'baby', 'la', 'vista'], 
['hasta', 'siempre', 'comandante', 'baby', 'la', 'siempre'], 
['siempre', 'comandante', 'baby', 'la', 'siempre']]

print ( len(texts)/sum([1.0 for i in texts if 'vista' in i]) )

text = ['hasta', 'la', 'vista', 'baby', 'la', 'vista', 'la']
print (calc_tf(text))







def calc_tf(docum):
    """
    parameters: docum - List of words
    returns: Counter object with TF for all words in docum
    """
    tf = collections.Counter(docum)
    for i in tf:
        tf[i] = tf[i]/float(len(docum))
    return tf

def calc_idf(word, corpus):
    """
    parameters: corpus - List of texts
                word - or which to calculate IDF
    returns: value, idf for word in corpus            
    """
    word_in_doc_count = sum([1.0 for i in corpus if word in i])
    idf = word_in_doc_count/len(corpus)
    return idf
        
def calc_tfidf(corpus):
    docs_list = []
    for doc in corpus:
        tf_idf_dict = {}
        tf = calc_tf(doc)
        for word in tf:
            tf_idf_dict[word] = tf[word] / calc_idf(word, corpus)
        docs_list.append(tf_idf_dict)
    return docs_list


texts = [['pasta', 'la', 'vista', 'baby', 'la', 'vista'], 
['hasta', 'siempre', 'comandante', 'baby', 'la', 'siempre'], 
['siempre', 'comandante', 'baby', 'la', 'siempre']]

print (calc_tfidf(texts))





    def compute_tf(text):
        #преобразуем входной список в каунтер
        tf_text = Counter(text)
        #используем генератор словарей для деления значения каждого элемента
        #в каунтере на общее число слов в тексте - т.е. длину списка слов.
        tf_text = {i: tf_text[i]/float(len(text)) for i in tf_text}
        return tf_text


def compute_tfidf(corpus):
    
    def compute_tf(text):
        tf_text = Counter(text)
        for i in tf_text:
            tf_text[i] = tf_text[i]/float(len(text))
        return tf_text
    
    def compute_idf(word, corpus):
        return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))
 
    documents_list = []
    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
        documents_list.append(tf_idf_dictionary)
    return documents_list

corpus = [[‘pasta’, ‘la’, ‘vista’, ‘baby’, ‘la’, ‘vista’],
[‘hasta’, ‘siempre’, ‘comandante’, ‘baby’, ‘la’, ‘siempre’],
[‘siempre’, ‘comandante’, ‘baby’, ‘la’, ‘siempre’]]

print compute_tfidf(corpus)


p = dict()
maximum = max(p, key=p.get)
print(maximum, p[maximum])




import nltk
nltk.download('stopwords')  # 1 time   or nltk.download()
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)


#Common stop words from online
stop_words = [
"a", "about", "above", "across", "after", "afterwards", 
"again", "all", "almost", "alone", "along", "already", "also",    
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", 
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]




# Train data
data = [["Chinese Beijing Chinese","0"],
        ["Chinese Chinese Shanghai","0"],
        ["Chinese Macao","0"],
        ["Tokyo Japan Chinese","1"]]

# P(word|class)=(word_count_in_class + 1)/(total_words_in_class+total_unique_words_in_corpus) 

# separate data on corpus and labels
corpus = []
labels = []
for text, label in data:
    corpus.append(text.split())
    labels.append(label)
classes = Counter(labels)
print (corpus)
print (labels)

print ("---")
for label in classes:
    print (label)
    print (classes[label])
    
print ("--- unique_words_in_corpus")
unique_words_in_corpus = Counter()
for doc in corpus:
    unique_words_in_corpus += Counter(doc)
print (len(unique_words_in_corpus))

print ("---")
for i in range(len(corpus)):
    print (len(corpus[i]))
    print (labels[i])
    
#total_words_in_class
print ("--- total_words_in_class")
total_words_in_class = defaultdict(int)
for i in range(len(corpus)):
    total_words_in_class[labels[i]] += len(corpus[i])
print (total_words_in_class)
print ("---")

# Conditional Probabilities
print ("--- words_in_class")
words_in_class = defaultdict(Counter)
for i in range(len(corpus)):
    words_in_class[labels[i]] += Counter(corpus[i])
print (words_in_class)
print (sum(words_in_class['1'].values()))
print (words_in_class['0']['Chinese'])

