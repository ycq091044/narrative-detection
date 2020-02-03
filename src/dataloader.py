import pandas as pd
import numpy as np
from utils import textProcess
import time

def postProcess(wordList):
        return ' '.join(np.sort(wordList.split(' ')))

def hawaiiData():
    # create corpus
    Corpus1 = ['earthquake', 'Hawaii', 'tourists', 'pacific', 'coast', \
            'witness', 'video', 'crack', 'damage', '2019', 'reason']
    Corpus2 = ['bomb', 'Russia', 'terroist', 'TNT', 'nuclear', 'duma', 'Putin', 'Soviet']
    Corpus3 = ['plate', 'drift', 'natrual', 'disaster', 'volcano', 'melt', 'magma', 'mountain']
    Corpus4 = ['curse', 'end', 'Voldemort', 'magic', 'black', 'mistery', 'religion', 'dragon']

    # number of online users
    UN1, UN2, UN3, UN4 = 100, 100, 100, 100

    # each user say Num of tweets
    Num = 10

    label = []
    tweet = []
    user = []
    # generate for Narrative1
    for i in range(UN1 * Num):
        user.append(i // Num)
        label.append(0)
        tweet.append(' '.join(np.random.choice(Corpus1*1 + Corpus2*0 + Corpus3*0 + Corpus4*0, np.random.randint(10, 20))))

    # generate for Narrative2
    for i in range(UN2 * Num):
        user.append(i // Num + UN1)
        label.append(1)
        tweet.append(' '.join(np.random.choice(Corpus1*1 + Corpus2*1 + Corpus3*0 + Corpus4*0, np.random.randint(10, 20))))

    # generate for Narrative3
    for i in range(UN3 * Num):
        user.append(i // Num + UN1 + UN2)
        label.append(2)
        tweet.append(' '.join(np.random.choice(Corpus1*1 + Corpus2*0 + Corpus3*1 + Corpus4*0, np.random.randint(10, 20))))
        
    # generate for Narrative4
    for i in range(UN4 * Num):
        user.append(i // Num + UN1 + UN2 + UN3)
        label.append(3)
        tweet.append(' '.join(np.random.choice(Corpus1*1 + Corpus2*0 + Corpus3*0 + Corpus4*1, np.random.randint(10, 20))))

    data = pd.DataFrame(np.hstack([np.array(user).reshape(-1, 1), np.array(tweet).reshape(-1, 1), np.array(label).reshape(-1, 1)]), \
                columns=['name', 'postTweet', 'label'])

    # postProcess
    data.postTweet = data.postTweet.apply(lambda x: postProcess(x))

    return data

def getData(pathD, pathK, kthreshold, uthreshold):
    if pathD == 'hawaii':
        return hawaiiData()
    else:
        data = pd.read_csv(pathD, sep='\t')
        print ('{} tweets and {} users'.format(len(data), len(data.name.unique())))
        return textProcess(data, pathK, kthreshold, uthreshold)

if __name__ ==  '__main__':
    print (hawaiiData())
