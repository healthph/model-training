import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score
from scipy.sparse import csr_matrix

######### CLEANING #############
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define stopwords
stopwords = set(stop_words_en)

# Load additional stopwords from 'tg.txt'
with open('tagalog_stop_words.txt', 'r') as f:
    additional_stopwords = f.read().splitlines()
stopwords.update(additional_stopwords)

stopwords.update([
    'wala', 'akong', 'ba', 'nung', 'talaga', 'pag', 'nang', 'de', 'amp', 'gym', 'coach',
    'lang', 'yung', 'kasi', 'naman', 'mo', 'di', 'si', 'nya', 'yun', 'im', 'will', 'sya', 'nga',
    'daw', 'eh', 'que', 'ug', 'e', 'man', 'jud', 'gi', 'oy',
    'ba', 'talaga', 'day', 'one', 'parang', 'know', 'wala', 'alam', 'tapos', 'pag', 'tao',
    'kayo', 'nung', 'us', 'now', 'natin', 'nasa', 'even', 'niyo', 'teaser', 'u', 'ma', 'yan'
])

def clean_text(text):
    if pd.notna(text):
        text = text.lower()
        text = re.sub(r'see\s+more', '', text, flags=re.IGNORECASE)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'([a-z])\1{1,}', r'\1', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\b\w\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
        text = ' '.join(filtered_tokens)

        return text
    return text

def clean_dataframe(df):
    cleaned_df = df.copy()
    cleaned_df['posts'] = cleaned_df['posts'].apply(clean_text)
    cleaned_df = cleaned_df.dropna(subset=['posts'])
    return cleaned_df


####################### MAPPINGS ##################


from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

def get_geopy_longlat(address):
    address_str = address
    if not address_str:
        return "-999, -999"
    geolocator = Nominatim(user_agent="https")
    try:
        location = geolocator.geocode(address_str)
        # Entire dictionary: location.raw
        return (location.latitude, location.longitude)
    except:
        return "-999, -999"


def get_geopy_reverse(longlat):
    if longlat == "-999, -999":
        #return {}
        return ""
    geolocator = Nominatim(user_agent="https")
    try:
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
        location = reverse(longlat, language='en', exactly_one=True)
        #return str(location.raw["address"])
        return location.raw["address"]
    except:
        #return {}
        return ""

def get_PH_Code(longlat):
  if longlat == "-999, -999":
    return "N/A"
  else:
    code = get_geopy_reverse(str(longlat))
    PH_code = code['ISO3166-2-lvl3']
    return PH_code

def get_lat(longlat):
  loc = str.split(longlat)
  lat = str(loc[0])
  late = lat[:-1]
  return late

def get_long(longlat):
  loc = str.split(longlat)
  long = loc[1]
  return long

def filter_location(province, region):
  address_str = str(province) + ", " + str(region)
  loc = get_geopy_longlat(address_str)
  if loc == "-999, -999":
    address_str = str(region)
    loc = get_geopy_longlat(address_str)
    if loc == "-999, -999":
      return "-999, -999"
    else:
      return filter(loc)
  else:
    return filter(loc)


def filter(loc):
  if loc[0] < 4.6666667 or loc[0] > 21.16666667: #lat
    return "-999, -999"

  elif loc[1] < 116.666666667 or loc[1] > 126.5666666666: #long
    return "-999, -999"

  else:
    return f"{loc[0]}, {loc[1]}"

############################ ANNOTATIONS ############

def annotatePosts(vectorizar,mlknn_c,post):
  vetorizar = vectorizar
  mlknn_classifier = mlknn_c
  post = [str(post)]
  post_tfidf = vetorizar.transform(post)
  predicted_post = mlknn_classifier.predict(post_tfidf)
  pred_post = predicted_post.toarray()
  #print(pred_post)
  annotate = []
  if pred_post[0][0] == 1:
    annotate.append("AURI")
  else:
    pass
  if pred_post[0][1] == 1:
    annotate.append("PN")
  else:
    pass
  if pred_post[0][2] == 1:
    annotate.append("TB")
  else:
    pass
  if pred_post[0][3] == 1:
    annotate.append("COVID")
  else:
    pass
  if annotate == []:
    annotate = "X"
  return annotate


####################### BINARY MATRIX ############
def get_AURI(annotation):
  a = str(annotation)
  if a == "X" or a == "x":
    return 0
  else:
    annotate = str.split(a)
    an = 0
    for i in annotate:
      if i == "AURI" or i == "AURI,":
        an = 1
      else:
        an = 0 + an
    return an

def get_PN(annotation):
  a = str(annotation)
  if a == "X" or a == "x":
    return 0
  else:
    annotate = str.split(a)
    an = 0
    for i in annotate:
      if i == "PN" or i == "PN,":
        an = 1
      else:
        an = 0 + an
    return an

def get_COVID(annotation):
  a = str(annotation)
  if a == "X" or a == "x":
    return 0
  else:
    annotate = str.split(a)
    an = 0
    for i in annotate:
      if i == "COVID" or i == "COVID,":
        an = 1
      else:
        an = 0 + an
    return an

def get_TB(annotation):
  a = str(annotation)
  if a == "X" or a == "x":
    return 0
  else:
    annotate = str.split(a)
    an = 0
    for i in annotate:
      if i == "TB" or i == "TB,":
        an = 1
      else:
        an = 0 + an
    return an


##### Read CSV File OF DATASET FOR TRAINING
csv_filename = "augmented_dataset.csv"
raw = pd.read_csv(csv_filename)
#dfa = pd.DataFrame(raw, columns = ["y","post"])

#CSV FILE SHOULD HAVE 
dfa = pd.DataFrame(raw, columns = ["annotate","posts"])

#annotation
dfa['AURI'] = dfa.apply(lambda x: get_AURI(x['annotate']), axis=1)
dfa['PN'] = dfa.apply(lambda x: get_PN(x['annotate']), axis=1)
dfa['TB'] = dfa.apply(lambda x: get_TB(x['annotate']), axis=1)
dfa['COVID'] = dfa.apply(lambda x: get_COVID(x['annotate']), axis=1)


X = dfa["posts"]
y = np.asarray(dfa[dfa.columns[2:]])

vetorizar = TfidfVectorizer(max_features=3000, max_df=0.85)
vetorizar.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_tfidf = vetorizar.transform(X_train)
X_test_tfidf = vetorizar.transform(X_test)

k_neighbors = 3  # Number of nearest neighbors

mlknn_classifier = MLkNN(k=k_neighbors)


mlknn_classifier.fit(X_train_tfidf, y_train)


################################# RAW DATA TO BE ANNOTATED BY MODEL ###############
raw_filename = "Raw_Data.csv"
raw = pd.read_csv(raw_filename)
#make sure it has header names for region and province and posts 
df = pd.DataFrame(raw, columns = ["region", "province", "posts"])

df['filtered_location'] = df.apply(lambda x: filter_location(x['province'], x['region']), axis=1)
df['PH_Code'] = df.apply(lambda x: get_PH_Code(x['filtered_location']), axis=1)
df['long'] = df.apply(lambda x: get_long(x['filtered_location']), axis=1)
df['lat'] = df.apply(lambda x: get_lat(x['filtered_location']), axis=1)

annotated_df = clean_dataframe(df)

annotated_df['annotations'] = annotated_df.apply(lambda x: annotatePosts(vetorizar, mlknn_classifier,x["posts"]), axis=1)

symptom_set = set(['cough', 'fever', 'ubo', 'sipon'])  # Define your existing set of words

annotated_df['extracted_words'] = annotated_df['posts'].apply(lambda x: ' '.join([word for word in x.split() if word in symptom_set]))
annotated_df.to_csv('./results/Result.csv')
