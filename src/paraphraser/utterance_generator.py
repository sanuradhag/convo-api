import re
import fileinput
import json
from textblob import TextBlob

def generate(line, slots):
    line = generate_syns(line)
    if len(slots) is not 0:
        line = append_slot(line, slots)
    line = line.rstrip()
    slots, utterances = expandLine( line[1:])
    seen = []
    for u in utterances:
        if u not in seen:
            # print(u)
            seen.append(u)
    return seen


def expandLine( line):
    slots, utterance = expandUtterance(line)
    return slots, ["{}".format(u) for u in utterance]


def expandUtterance(utterance):
    slots, utterance = collectSlots(utterance)
    return slots, [alternative for optional in expandOptionals(utterance) for alternative in
                   expandAlternatives(optional)]


def expandOptionals(utterance):
    if '[' in utterance:
        return expandOptionals(re.sub("\[([^\[\]]+)\]", "\g<1>", utterance, 1)) + expandOptionals(
            re.sub("\[([^\[\]]+)\]", "", utterance, 1))
    return [utterance]


def expandAlternatives(utterance):
    m = re.search(r'\(([^\(\)]+)\)', utterance)
    if not m:
        return [utterance]
    match = m.group(0)
    words = m.group(1).split("|")
    return [a for w in words for a in expandAlternatives(utterance.replace(match, w))]


def collectSlots(utterance):
    p = re.compile("\{([\w.]+):(\w+)\}")
    return dict([(a, b) for b, a in p.findall(utterance)]), p.sub(lambda m: "{{{}}}".format(m.group(2)), utterance)


def generate_syns(sentence):
    wiki = TextBlob(sentence)
    # es = wiki.translate(to='es')
    # en = es.translate(to='en')

    tags = wiki.tags

    map = []
    for t in tags:
        word, tag = t[0],t[1]
        word_dict = {'word': word, 'syns': ''}
        if tag == 'NN' or tag == 'NNS' or tag == 'NNP':
            synonyms = list()
            str = '('
            for synset in word.synsets:
                for lemma in synset.lemmas():
                    synonyms.append(lemma.name())
            for i in range(3):
                str += synonyms[i] + '|'
            str = str[:-1]
            str += ')'
            word_dict['syns'] = str
            map.append(word_dict)
        else:
            map.append(word_dict)

    output = ''
    for i in range(len(map)):
        mapping = map[i]
        syns = mapping['syns']
        if syns is not '':
            output += ' ' + mapping['syns']
        else:
            output += ' ' + mapping['word']

    return output

def append_slot(sentence, slots):
    for i in range(len(slots)):
        sentence += ' {'+slots[i]['type']+':'+slots[i]['name']+'}'
    return sentence