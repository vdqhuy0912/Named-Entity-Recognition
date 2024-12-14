import numpy as np

def readfile(filename, *, encoding="UTF8"):
    '''
    Đọc file và trả về định dạng:
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ... ]
    '''
    with open(filename, mode='rt', encoding=encoding) as f:
        sentences = []
        sentence = []
        for line in f:
            line = line.strip()  # Loại bỏ khoảng trắng và ký tự xuống dòng
            if len(line) == 0 or line.startswith('-DOCSTART'):
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            splits = line.split(' ')
            sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def addCharInformation(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0], chars, data[1]]
    return Sentences

def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup.get(casing, caseLookup['other'])

def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx, max_char_len=52):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, chars, label in sentence:
            # Word Index
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx

            # Casing Index
            casing_idx = getCasing(word, case2Idx)

            # Character Indices
            char_idx = [char2Idx.get(c, char2Idx['UNKNOWN']) for c in chars]
            if len(char_idx) < max_char_len:
                char_idx += [char2Idx['PADDING']] * (max_char_len - len(char_idx))
            else:
                char_idx = char_idx[:max_char_len]

            # Label Index
            label_idx = label2Idx[label]

            wordIndices.append(wordIdx)
            caseIndices.append(casing_idx)
            charIndices.append(char_idx)
            labelIndices.append(label_idx)

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    return dataset

def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len