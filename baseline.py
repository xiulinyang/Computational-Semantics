import collections
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from preprocess_and_basic_info import get_word_tag, flattern
from preprocess_and_basic_info import train_sent, dev_sent, train_word, dev_word, \
  train_sent_x, train_sent_y, dev_sent_x, dev_sent_y, one_tag_word, multi_tag_word
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def create_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("-word", "--word feature", dest='word', action='store_true',
                      help="Use word feature.")
  parser.add_argument("-morph", "--morphological feature", dest='morph', action="store_true",
                      help="Use morphological features + word features")
  parser.add_argument("-suffix","--suffix feature", dest='suffix', action="store_true",
                      help="Use morphological features + word features + prefix and suffix features")
  args = parser.parse_args()
  return args




"""# Dummy classifier
I will build a dummy classifier which will only return the most frequent tag for the word
"""

# Check the data distribution

def dummy_classifier():
  # check how many tokens in the data have only one tag
  one_tag = get_word_tag(train_word, one_tag_word)
  # check how many tokens in the data have more than one tag
  more_tag = get_word_tag(train_word, multi_tag_word)
  # find the most frequent tag for those tokens
  freq_list = collections.Counter(more_tag).most_common(len(more_tag))
  most_frequent_multi_word =[]
  most_frequent_multi_tag =[]
  count =0
  for word in freq_list:
    if word[0][0] not in most_frequent_multi_word:
      most_frequent_multi_word.append(word[0][0])
      most_frequent_multi_tag.append((word[0][0], word[0][1]))
      count += word[1]
  predict_set = one_tag + most_frequent_multi_tag
  # performance for the training set
  acc_train = (len(one_tag)+count)/len(flattern(train_sent_x))
  # performance for the development set
  predict_dummy = []
  total_dev_words =[]
  for sent in dev_word:
    for word in sent:
      total_dev_words.append(word)
      if word in predict_set:
        predict_dummy.append(word)
  predict_dummy = []
  total_dev_words = []
  for sent in dev_word:
    for word in sent:
      total_dev_words.append(word[1])
      if word in predict_set:
        predict_dummy.append(word[1])
      elif word not in predict_set:
        predict_dummy.append('NIL')
  # print(len(total_dev_words), len(predict_dummy))
  return acc_train, classification_report(total_dev_words, predict_dummy, digits=3)

args = create_arg_parser()
def extract_features(sentence, index, args):
  if args.word:
    return {
      'word': sentence[index]
    }
  if args.morph:
    return {
      'word': sentence[index],
      'is_first': index == 0,
      'is_last': index == len(sentence) - 1,
      'is_capitalized': sentence[index][0].upper() == sentence[index][0],
      'is_all_caps': sentence[index].upper() == sentence[index],
      'is_all_lower': sentence[index].lower() == sentence[index],
      'prev_word': '' if index == 0 else sentence[index - 1],
      'next_word': '' if index > len(sentence) - 2 else sentence[index + 1],
      'has_tilder': '~' in sentence[index],
      'is_numeric': sentence[index].isdigit(),
      'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }
  if args.suffix:
    return {
      'word': sentence[index],
      'is_first': index == 0,
      'is_last': index == len(sentence) - 1,
      'is_capitalized': sentence[index][0].upper() == sentence[index][0],
      'is_all_caps': sentence[index].upper() == sentence[index],
      'is_all_lower': sentence[index].lower() == sentence[index],
      'prev_word': '' if index == 0 else sentence[index - 1],
      'next_word': '' if index > len(sentence) - 2 else sentence[index + 1],
      'prefix-1': sentence[index][0],
      'prefix-2': sentence[index][:2],
      'prefix-3': sentence[index][:3],
      'suffix-1': sentence[index][-1],
      'suffix-2': sentence[index][-2:],
      'suffix-3': sentence[index][-3:],
      'has_tilder': '~' in sentence[index],
      'is_numeric': sentence[index].isdigit(),
      'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def transform_to_dataset(tagged_sentences):
  X, y = [], []
  for sentence, tags in tagged_sentences:
    sent_word_features, sent_tags = [],[]
    for index in range(len(sentence)):
        sent_word_features.append(extract_features(sentence, index, args)),
        sent_tags.append(tags[index])
    X.append(sent_word_features)
    y.append(sent_tags)
  return X, y


if __name__ =='__main__':

  acc_train, acc_dev = dummy_classifier()
  print('The accuracy of the dummy classifier is: ', acc_train, '\n and a detailed report for this classifier on '
                                                                'a development set is ', acc_dev)
  X_train, y_train = transform_to_dataset(train_sent)
  X_dev, y_dev = transform_to_dataset(dev_sent)
  X_train = flattern(X_train)
  y_train = flattern(y_train)
  X_dev = flattern(X_dev)
  y_dev = flattern(y_dev)
  dict_vec = DictVectorizer(sparse=False)
  dict_vec.fit(X_train)
  X_train = dict_vec.transform(X_train)
  X_dev = dict_vec.transform(X_dev)

  print('Begin the training!')
  clf = DecisionTreeClassifier()
  clf.fit(X_train, y_train)
  print('Finish the training!')
  y_pred = clf.predict(X_dev)

  print('The classification report of the baseline is: \n')
  print(classification_report(y_dev, y_pred, digits=3))




