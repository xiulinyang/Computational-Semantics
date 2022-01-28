import sklearn
from sklearn.feature_extraction import DictVectorizer
from nltk.tag import tnt
import argparse
import warnings
from sklearn_crfsuite import CRF
warnings.filterwarnings('ignore')
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import spacy
from spacy.tokens import Doc
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from preprocess_and_basic_info import get_word_tag, flattern
from preprocess_and_basic_info import train_sent, dev_sent, train_word, dev_word, \
  train_sent_x, train_sent_y, dev_sent_x, dev_sent_y, one_tag_word, multi_tag_word



def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pos1", "--use 1 pos", dest= "pos1", action="store_true",
                        help="Use the pos before and after the token")
    parser.add_argument("-pos2", "--use 2 pos", dest="pos2", action = "store_true",
             help = "Use two pos before and after the token")
    parser.add_argument("-pos3", "--use 3 pos", dest="pos3", action = "store_true",
                        help = "Use three pos before and after the token")
    parser.add_argument("-seq", "--delete sequence info", dest = "seq", action="store_true",
                        help="Delete the sequential information in the feature")
    args = parser.parse_args()
    return args


'''PART OF SPEECH'''

nlp =spacy.load("en")
def get_pos(train_sent_x):
  train_pos =[]
  for sent in train_sent_x:
    sent_pos =[]
    doc = Doc(nlp.vocab, words=sent)
    doc = nlp.tagger(doc)
    for word in doc:
      sent_pos.append((word.text, word.pos_))
    train_pos.append(sent_pos)
  return train_pos


def combine_pos(train_pos, train_sent):
  pos =[]
  for sent in train_pos:
    sent_poss =[]
    for word in sent:
      sent_poss.append(word[1])
    pos.append(sent_poss)

  train_with_pos =[]
  for i, sent in enumerate(train_sent):
    train_with_pos.append((sent[0], sent[1], pos[i]))

  return train_with_pos



args = create_argparser()

def extract_features(sentence, pos, index):
    if args.pos1:
        return {
            'word':sentence[index],
            'is_first':index==0,
            'is_last':index ==len(sentence)-1,
            'pos': pos[index],
            'is_capitalized':sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'prev_word':'' if index == 0 else sentence[index-1],
            'next_word':'' if index > len(sentence)-2 else sentence[index+1],
            'prev_pos': '' if index == 0 else pos[index - 1],
            'next_pos': '' if index > len(sentence) - 2 else pos[index + 1],
            'has_tilder': '~' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }
    if args.pos2:
        return {
          'word':sentence[index],
          'is_first':index==0,
          'is_last':index ==len(sentence)-1,
          'pos': pos[index],
          'is_capitalized':sentence[index][0].upper() == sentence[index][0],
          'is_all_caps': sentence[index].upper() == sentence[index],
          'is_all_lower': sentence[index].lower() == sentence[index],
          'prev_word':'' if index == 0 else sentence[index-1],
          'next_word':'' if index > len(sentence)-2 else sentence[index+1],
          'prev_pos': '' if index == 0 else pos[index - 1],
          'next_pos': '' if index > len(sentence) - 2 else pos[index + 1],
          'prev_2_pos': '' if index < 2 else pos[index - 2],
          'next_2_pos': '' if index > len(sentence) - 3 else pos[index + 2],
          'has_tilder': '~' in sentence[index],
          'is_numeric': sentence[index].isdigit(),
          'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
      }
    if args.pos3:
        return {
              'word':sentence[index],
              'is_first':index==0,
              'is_last':index ==len(sentence)-1,
              'pos': pos[index],
              'is_capitalized':sentence[index][0].upper() == sentence[index][0],
              'is_all_caps': sentence[index].upper() == sentence[index],
              'is_all_lower': sentence[index].lower() == sentence[index],
              'prev_word':'' if index == 0 else sentence[index-1],
              'next_word':'' if index > len(sentence)-2 else sentence[index+1],
              'prev_pos': '' if index == 0 else pos[index - 1],
              'next_pos': '' if index > len(sentence) - 2 else pos[index + 1],
              'prev_2_pos': '' if index < 2 else pos[index - 2],
              'next_2_pos': '' if index > len(sentence) - 3 else pos[index + 2],
              'prev_3_pos': '' if index < 3 else pos[index - 3],
              'next_3_pos': '' if index > len(sentence) - 4 else pos[index + 3],
              'has_tilder': '~' in sentence[index],
              'is_numeric': sentence[index].isdigit(),
              'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
          }
    if args.seq:
        return {
        'word':sentence[index],
        'pos': pos[index],
        'is_capitalized':sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'has_tilder': '~' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]}

    if args.seq:
        return {
        'word':sentence[index],
        'pos': pos[index],
        'is_capitalized':sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'has_tilder': '~' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]}


def transform_to_dataset(tagged_sentences):
  X, y = [], []
  for sentence, tags, pos in tagged_sentences:
    sent_word_features, sent_tags = [],[]
    for index in range(len(sentence)):
        sent_word_features.append(extract_features(sentence, pos, index)),
        sent_tags.append(tags[index])
    X.append(sent_word_features)
    y.append(sent_tags)
  return X, y


def tnt_training(train_word, dev_word):
    tnt_tagging = tnt.TnT()
    tnt_tagging.train(train_word)
    eval = tnt_tagging.evaluate(dev_word)
    return eval





def transform_to_linear(xx_train, yy_train, xx_dev, yy_dev):
    X_train = flattern(xx_train)
    y_train = flattern(yy_train)
    X_dev = flattern(xx_dev)
    y_dev = flattern(yy_dev)
    dict_vec = DictVectorizer(sparse =False)
    dict_vec.fit(X_train)
    X_train = dict_vec.transform(X_train)
    X_dev = dict_vec.transform(X_dev)
    return X_train, y_train, X_dev, y_dev


def experiment_with_pos(X_train,y_train, X_dev, y_dev):
    clf_tree = DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    y_pred = clf_tree.predict(X_dev)
    return classification_report(y_dev, y_pred, digits=3)


def experiment_with_rf(X_train,y_train, X_dev, y_dev):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    return classification_report(y_dev, y_pred, digits=3)


def experiment_with_svm(X_train,y_train, X_dev, y_dev):
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    return classification_report(y_dev, y_pred, digits=3)


train_pos = get_pos(train_sent_x)
dev_pos = get_pos(dev_sent_x)
train_with_pos = combine_pos(train_pos, train_sent)
dev_with_pos = combine_pos(dev_pos, dev_sent)
X_train, y_train = transform_to_dataset(train_with_pos)
X_dev, y_dev = transform_to_dataset(dev_with_pos)

def experiment_with_crf(X_train, y_train, X_dev):
    classifier_crf = CRF(
        algorithm='lbfgs',
        c1=0.026,
        c2=0.008,
        max_iterations=100,
        all_possible_transitions=True
    )
    classifier_crf.fit(X_train, y_train)
    labels = list(classifier_crf.classes_)
    y_pred=classifier_crf.predict(X_dev)
    return y_pred

if __name__ == '__main__':

    X_train_linear,y_train_linear, X_dev_linear,y_dev_linear = transform_to_linear(X_train, y_train,X_dev, y_dev)
    print('The classification report of Decision Tree: \n')
    print(experiment_with_pos(X_train_linear, y_train_linear, X_dev_linear, y_dev_linear))
    print('The classification report of SVM: \n')
    print(experiment_with_svm(X_train_linear, y_train_linear, X_dev_linear, y_dev_linear))
    print('The classification report of Random Forest: \n')
    print(experiment_with_rf(X_train_linear, y_train_linear, X_dev_linear, y_dev_linear))

    print('The accuracy of tnt is', tnt_training(train_word, dev_word))
    print('The classification report of CRF is: \n')
    pred_crf = flattern(experiment_with_crf(X_train, y_train, X_dev))
    y_dev_crf = flattern(y_dev)
    print(classification_report(y_dev_crf, pred_crf, digits=3))
