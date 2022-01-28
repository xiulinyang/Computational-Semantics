import collections
import matplotlib.pyplot as plt


def get_doc_sent(directory):
    doc = open(directory).readlines()
    all = []
    sentence = []
    for i, x in enumerate(doc):
        if x.startswith(('# ', '###')) == True:
            continue
        elif len(x) < 2:
            all.append(sentence)
            sentence = []
        else:
            sentence.append(x)
    return all


# In the output, each sentence is mapped with its corresponding tags


def get_data_sent(doc):
    all = []
    for sent in doc:
        tok = []
        sem = []
        for line in sent:
            line = line.split('\t')
            if len(line) == 7:
                tok.append(line[0])
                sem.append(line[3])
        all.append((tok, sem))
    return all


# get the data in the form that each word and their corresponding tag are in one tuple.
def get_doc_word(directory):
    doc = open(directory).readlines()
    all = []
    sentence = []
    for i, x in enumerate(doc):
        if x.startswith(('# ', '###')) == True:
            continue
        elif len(x) < 2:
            all.append(sentence)
            sentence = []
        else:
            sentence.append(x)
    return all


def get_data_word(doc):
    all = []
    for sent in doc:
        tok_sem = []
        for line in sent:
            line = line.split('\t')
            if len(line) == 7:
                tok_sem.append((line[0], line[3]))
        all.append((tok_sem))
    return all


# only applicable to data divided by sentences, i.e. get_data_sentence
def split_data(data):
    word = []
    label = []
    for sent in data:
        word.append(sent[0])
        label.append(sent[1])
    return word, label


# coverting the nested list into one flat list
def flattern(list):
    flattern_list = []
    for element in list:
        flattern_list += element
    return flattern_list

# check the unique tokens in the data
def unique_tokens(train_sent_x):
    unique_words = []
    for sent in train_sent_x:
        for word in sent:
            unique_words.append(word)
    return set(unique_words)

# check if the multiple tags are correctly marked
def check_tag(train_word, token, tag):
  result =[]
  pattern = (token, tag)
  for sent in train_word:
    if pattern in sent:
      result.append(sent)
  return result

# define a function to get a list of (word, tag) tuple in which the word is in a list
def get_word_tag(train_word, list):
  result =[]
  for sent in train_word:
    for word in sent:
      if word[0] in list:
        result.append(word)
  return result


if __name__ == '__main__':
    dev_sent = get_data_sent(get_doc_sent('dev.txt'))
    train_sent = get_data_sent(get_doc_sent('train.conll.txt'))

    dev_word = get_data_word(get_doc_sent('dev.txt'))
    train_word = get_data_word(get_doc_sent('train.conll.txt'))

    train_sent_x, train_sent_y = split_data(train_sent)
    dev_sent_x, dev_sent_y = split_data(dev_sent)

    """ Data basic information """

    # how many sentences there are in the training set
    print('There are altogether', len(train_sent), 'sentences in the training set.')

    # how many words there are in the training set
    print('There are altogether', len(flattern(train_sent_x)), 'tokens in the training set.')

    """### Tag information"""

    # how many unique tokens in the training set

    print('There are ', len(unique_tokens(train_sent_x)), 'unique tokens.')

    # how many unique token-tag patterns
    flattern_train = flattern(train_word)
    set_flattern_train = set(flattern_train)
    print('There are', len(set_flattern_train), 'unique token-tag patterns.')

    # among those unique token-tag patterns, how many token are annotated with multiple tags
    unique_word_set = []
    for word in set_flattern_train:
        unique_word_set.append(word[0])

    freq = collections.Counter(unique_word_set)
    # details of the mapping data
    interesting_words = []
    multi_tag_word = []
    one_tag_word = []
    for key, value in freq.items():
        if value > 3:
            interesting_words.append(key)
        if value > 1:
            multi_tag_word.append(key)
        if value == 1:
            one_tag_word.append(key)

    # how many tokens that could be annotated by more than one tag
    print('There are', len(set(multi_tag_word)), 'tokens that could be annotated by more than one tag.')
    # how many tokens that could be annotated by one tag
    print('There are', len(one_tag_word), 'tokens that only have one tag')
    # check those 'interesting' words
    interesting_words_tag = get_word_tag(train_word, interesting_words)
    print('The following tokens are annotated by over 3 tags:\n',set(interesting_words_tag))

    # check a few examples
    print('In the train_word list, the pattern(, , BUT) is found in the following sentences:\n',
          check_tag(train_word, ',', 'BUT'))
    print('\n')
    print('In the train_word list, the pattern(, , REL) is found in the following sentences:\n',
          check_tag(train_word, ',', 'REL'))


    # visualize the distribution of tags
    y_list = flattern(train_sent_y)
    sem = collections.Counter(y_list)
    sem = sem.most_common(len(sem))
    x_val = [val[0] for val in sem]
    y_val = [val[1] for val in sem]
    plt.bar(x_val, y_val, label='train')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=3)
    plt.show

    print('There are altogher ', len(set(y_list)), 'tags.\n Their distributions are:\n', sem)