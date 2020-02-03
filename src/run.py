from dataloader import getData
from utils import decompostition, Mmodule, Smodule, getKeyMatrix
from sklearn.metrics import accuracy_score
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import argparse

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--pathA', default='hawaii',
                    help='path of adjacency relation')
parser.add_argument('--pathD', default='hawaii',
                    help='path of main data file')
parser.add_argument('--pathK', default='hawaii',
                    help='path of keywordList')
parser.add_argument('--fastmode', default='N',
                    help='use M-module or not (Y or N)')
parser.add_argument('--seed', type=int, default=150, help='Random seed.')
parser.add_argument('--l1', type=float, default=0.001,
                    help='coefficient of l1-norm')
parser.add_argument('--l2', type=float, default=0.001,
                    help='coefficient of l2-norm')
parser.add_argument('--N', type=int, default=4,
                    help='dimension of beliefs.')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of iterations')
parser.add_argument('--process', type=int, default=40,
                    help='number of available processes')
parser.add_argument('--kthreshold', type=int, default=5,
                    help='the number of minimun keywords contained in processed text')
parser.add_argument('--uthreshold', type=int, default=3,
                    help='the mininum frequency of user occurance')


args = parser.parse_args()

def getResult(mode, tweetMap, data, X, N, l1, l2, epochs):
    # plt.figure(figsize=(5, 4))
    U, M, loss, B  = decompostition(X, mode, N, l1, l2, epochs)
    # plt.plot(loss)
    # plt.title(mode + ' Loss')

    tempMap = []
    result = []
    for tweet in data.postTweet:
        tempMap.append(M[tweetMap[tweet], :])
        result.append(M[tweetMap[tweet], :].tolist())
    
    return np.argmax(tempMap, axis=1), M

def scoreResult(pre, gt, mode):

    Map = {'0': 0, 
        '1': 1,
        '2': 2,
        '3': 3}

    JudgeList = []
    if mode == 'BSMF':
        for permu in permutations([1, 2, 3]):
            Map2 = {0: 0, 1: permu[1], 2: permu[2], 3: permu[0]}
            tempTarget = [Map2[Map[t]] for t in gt]
            JudgeList.append(accuracy_score(pre, tempTarget))
    else:
        for permu in permutations([0, 1, 2, 3]):
            Map2 = {0: permu[0], 1: permu[1], 2: permu[2], 3: permu[3]}
            tempTarget = [Map2[Map[t]] for t in gt]
            JudgeList.append(accuracy_score(pre, tempTarget))
    return JudgeList

def showNarrative(tweetMap, data, M, N):
    delta = 0.01
    sigma = 3

    with open('../output/output.narrative', 'w') as outfile:
        print ('N = {}'.format(N))
        for i in range(N):
            print ('Narrative', i+1); print ('Narrative', i+1, file=outfile)
            print ('---------------------------'); print ('---------------------------', file=outfile)
            tempMap = []

            for tweet in data.postTweet:
                tempMap.append(M[tweetMap[tweet], i])
            tempMap = np.argsort(-np.array(tempMap))

            keyPocket = set()
            s = 0

            for i, (tweet, postTweet) in enumerate(data[['rawTweet', 'postTweet']].iloc[tempMap].values):
                if s >= sigma:
                    break
                # jaccard
                key = postTweet.split()
                if len(set(key) - keyPocket) / (len(keyPocket) + 1) >= delta:
                    print (tweet); print (tweet, file=outfile)
                    print ('-------------'); print ('-------------', file=outfile)
                    s += 1
                for item in key:
                    keyPocket.add(item)
            print (); print(file=outfile)

def run(pathA=args.pathA, pathD=args.pathD, pathK=args.pathK, fastmode=args.fastmode, \
    N=args.N, l1=args.l1, l2=args.l2, epochs=args.epochs, K=args.process):

    np.random.seed(args.seed)

    # pre-processing
    data = getData(pathD=args.pathD, pathK=args.pathK, kthreshold=args.kthreshold, uthreshold=args.uthreshold)
    A = Smodule(data.name.unique().tolist(), pathA=args.pathA)
    userMap, tweetMap, userKey, tweetKey, userTweet, userTweet2 = getKeyMatrix(data)
    if fastmode == 'N':
        userTweet = Mmodule(userTweet, tweetKey, K)
        X = A @ userTweet
    elif fastmode == 'Y':
        X = A @ userTweet2

    if 'label' in data:
        for mode in ['NMF', 'NMTF', 'BSMF']:
            pre, _ = getResult(mode, tweetMap, data, X, N, l1, l2, epochs)
            with open('./label/' + mode + '.label', 'w') as outfile:
                for i, l in enumerate(pre):
                    print ('{}\t{}'.format(i, l), file = outfile)
            JudgeList = scoreResult(pre, data.label.tolist(), mode)
            print (mode, 'accuracy', max(JudgeList))
    else:
        print ()
        _, M = getResult('BSMF', tweetMap, data, X, N, l1, l2, epochs)
        showNarrative(tweetMap, data, M, N)
    
    # plt.show()

run()
