import pandas as pd
import numpy as np
import time
from multiprocessing import Process
from multiprocessing import Manager
from itertools import combinations
from numpy import linalg as LA
import re

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=5)

# method by Jianing
def strip_symbol(words):
    """
    Delete all the symbol inside a word
    :param word
    """
    return re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',words)

# tokenize filter
def lenFilter(word):
    return len(word) >= 2

# tokenize
def tokenize(text, stopwords = [], keyword = []):
    # get rid of URL
    original_text = str(text).lower()
    tok = original_text.split(' ')
    text = u''
    for x in tok:
        if len(keyword) > 0:
            if x not in keyword: continue
        elif len(stopwords) > 0:
            if len(x) == 0: continue
            elif x[0:4] == 'http' or x[0:5] == 'https': continue
            elif x[0] == '@': continue
            elif x in stopwords: continue
        text = text + ' ' + x
    translate_to = u' '

    word_sep = u" ,.?:;'\"/<>`!$%^&*()-=+~[]\\|{}()\n\t" \
        + u"©℗®℠™،、⟨⟩‒–—―…„“”–――»«›‹‘’：（）！？=【】　・" \
        + u"⁄·† ‡°″¡¿÷№ºª‰¶′″‴§|‖¦⁂❧☞‽⸮◊※⁀「」﹁﹂『』﹃﹄《》―—" \
        + u"“”‘’、，一。►…¿«「」ー⋘▕▕▔▏┈⋙一ー।;!؟"
    word_sep = u'#' + word_sep
    translate_table = dict((ord(char), translate_to) for char in word_sep)
    tokens = text.translate(translate_table).split(' ')
    return ' '.join(sorted(list(filter(lenFilter, tokens))))

# from rawTweet to clean keyword text
def textProcess(data, pathK, kthreshold, uthreshold):

    stopwords = []
    keyword = []
    if pathK == 'N':
        # get stopwords
        with open('../sample/stopwords_en.txt', 'r') as infile:
            for word in infile.readlines():
                stopwords.append(word[:-1])

        data['postTweet'] = data.rawTweet.parallel_apply(lambda x: tokenize(x, stopwords=stopwords, keyword=[]))
    else:
        # get stopwords
        with open('../processed/keyword.txt', 'r') as infile:
            for word in infile.readlines():
                keyword.append(word[:-1])
        data['postTweet'] = data.rawTweet.parallel_apply(lambda x: tokenize(x, stopwords=[], keyword=keyword))
    
    # number of keywords >= 5
    data['keyN'] = data.postTweet.apply(lambda x: len(x.split()))
    data = data[data.keyN >= kthreshold]

    userDict = dict()
    for u in data.name.values:
        try:
            userDict[u] += 1
        except:
            userDict[u] = 1

    pickedPopUsers = np.array(list(userDict.keys()))[np.where(np.array(list(userDict.values())) >= uthreshold)]
    data = data[data.name.isin(pickedPopUsers)]
    data.reset_index(drop=True, inplace=True)

    return data

# user-index map
def getUserMap(data):
    userMap = dict()
    for i, user in enumerate(data.name.unique()):
        userMap[user] = i
    return userMap

 # tweet-index map
def getTweetMap(data):
    tweetMap = dict()
    for i, tweet in enumerate(data.postTweet.unique()):
        tweetMap[tweet] = i
    return tweetMap

# construct user-tweet matrix
def bimatrix(userMap, tweetMap, data):
    userTweet = np.zeros((len(userMap), len(tweetMap)))
    for user, tweet in data[['name', 'postTweet']].iloc[::-1].values:
        userTweet[userMap[user], tweetMap[tweet]] += 1
    return userTweet

# construct keyword list
def keyList(data):    
    keywordList = []
    for tweet in data.postTweet:
        keywordList += tweet.split()
    keywordList = set(keywordList)
    print ('keyword corpus:', len(keywordList))
    return keywordList

# get user-key matrix
def returnDistUser(data, keywordList, user):
    tempKey = []
    tempCount = dict.fromkeys(keywordList, 0)
    for tweet in data[data.name == user].postTweet:
        tempKey += tweet.split()
        
    for word in set(tempKey):
        tempCount[word] = tempKey.count(word)
    return list(tempCount.values())

# construct user-keyword Matrix
def getUserKey(userMap, data, keywordList):
    userKey = pd.DataFrame(userMap.keys(), columns=['name'])
    tic = time.time()
    userKey['dist'] = userKey.name.parallel_apply(lambda x: returnDistUser(data, keywordList, x))
    userKey = np.array(userKey.dist.values.tolist())
    print ('user Key success. take times,', time.time() - tic)
    return userKey

# get tweet-key matrix
def returnDistTweet(keywordList, tweet):
    tempKey = tweet.split()
    tempCount = dict.fromkeys(keywordList, 0)
        
    for word in set(tempKey):
        tempCount[word] = tempKey.count(word)
    return list(tempCount.values())

# construct tweet-keyword Matrix
def getTweetKey(tweetMap, keywordList):
    tweetKey = pd.DataFrame(tweetMap.keys(), columns=['tweet'])
    tic2 = time.time()
    tweetKey['dist'] = tweetKey.tweet.parallel_apply(lambda x: returnDistTweet(keywordList, x))
    tweetKey = np.array(tweetKey.dist.values.tolist())
    print ('tweet Key success. take times,', time.time() - tic2)
    return tweetKey

# First Step Process
def getKeyMatrix(data):
    # get Maps
    userMap, tweetMap = getUserMap(data), getTweetMap(data)
    # get biMatrix
    userTweet = bimatrix(userMap, tweetMap, data)
    # get keyList
    keywordList = keyList(data)
    # userKey
    userKey = getUserKey(userMap, data, keywordList)
    # tweetKey
    tweetKey = getTweetKey(tweetMap, keywordList)

    # normalize by 2-norm
    userKey = userKey / (userKey ** 2).sum(axis=1).reshape(-1, 1) ** 0.5
    tweetKey = tweetKey / (tweetKey ** 2).sum(axis=1).reshape(-1, 1) ** 0.5

    userTweet2 = userKey @ tweetKey.T

    print ('# of users', userTweet2.shape[0], ', # of tweets', userTweet2.shape[1])

    return userMap, tweetMap, userKey, tweetKey, userTweet, userTweet2

# interpolation function
def phi(nz_index, tweetKey, index, r):
    if index in nz_index:
        return 1
    s = 0
    for i in nz_index:
        s += np.exp(- r * np.linalg.norm(np.array(tweetKey[index, :]) - np.array(tweetKey[i, :]), 2) ** 2) / 4
    if s < 0.2:
        return 0
    else:
        return s

# def phi2(nz_index, tweetKey, index, r):
#     temp = np.ones(len(z_index))
#     print (temp.shape)
#     for j in index:
#         if j % 1000 == 0:
#             print (j, end=' ')
#         if j in nz_index:
#             continue
#         s = 0
#         for i in nz_index:
#             s += np.exp(- r * np.linalg.norm(np.array(tweetKey[index, :]) - np.array(tweetKey[i, :]), 2)**2)
#         if s < 0.05:
#             temp[j] = 0
#         else:
#             temp[j] = s
#     print ()
#     return temp.reshape(1, -1)
    
# son-process function
def interpolation(result, userTweet, tweetKey, k, K):
    for i in range(userTweet.shape[0]):
        if i % K == k:
            print ('process', k, '{} / {}'.format(i, userTweet.shape[0]))
            nz_index = np.where(userTweet[i, :] > 0)[0]
            index = np.arange(userTweet.shape[1])
            result[i] = np.vectorize(phi, \
                excluded=['nz_index', 'tweetKey'])(nz_index=nz_index, tweetKey=tweetKey, index=index, r=2)

# Message Similarity (M-Module): Second Step
def Mmodule(userTweet, tweetKey, K): 
    """
    K assigns the number of processes
    """
    manager = Manager()
    result = manager.dict()

    plist = []
    for k in range(K):
        temp = Process(target=interpolation, args=(result, userTweet, tweetKey, k, K))
        plist.append(temp)
        
    for i in plist:
        i.start()
    for i in plist:
        i.join()  
    
    userTweet = []
    for _, j in sorted(dict(result).items(), key=lambda x: x[0]):
        userTweet.append(j)
    userTweet = np.array(userTweet)

    return userTweet


# Social Graph Convolution (S-Module): Third Step
def Smodule(nameList, pathA):
    if pathA == 'hawaii':
        adjTable = np.zeros((400, 400))
        A = np.array(adjTable) + np.diag(np.ones(400))
        A = A / A.sum(axis=1).reshape(-1,1)
        A = (A + np.diag(np.ones(A.shape[0]))) / 2

    else:
        friend = pd.read_csv(pathA, sep='\t')

        adjTable = pd.DataFrame(columns = nameList)
        for user in nameList:
            adjTable.at[user] = np.zeros(len(nameList))

        for u1, u2 in friend.values:
            if (u1 in nameList) and (u2 in nameList):
                adjTable.at[u1, u2] += 1

        A = np.array(adjTable) + np.diag(np.ones(adjTable.shape[0]))
        A = A / A.sum(axis=1).reshape(-1,1)
        A = (A + np.diag(np.ones(A.shape[0]))) / 2

    return A

# define the loss
def lossFunc(X, U, B, M, lambda1, lambda2):
    loss1 = LA.norm(X - U @ B @ M.T, 'fro')
    loss2 = lambda2 * (LA.norm(U, 2) + LA.norm(M, 2))
    return loss1 + loss2

# matrix factorization: Final Step
def decompostition(X, mode, N, lambda1, lambda2, epochs):

    # User-belief (U) and Message-belief (M) matrix
    U = np.random.rand(X.shape[0], N)
    M = np.random.rand(X.shape[1], N)

    if mode == 'BSMF':
        # B = np.array([[1, 0, 0, 0],
        #               [1, 1, 0, 0],
        #               [1, 0, 1, 0],
        #               [1, 0, 0, 1]])
        B = np.eye(N) + np.array(np.array([[0]] + \
            [[1] for i in range(N-1)])) * np.array([1] + [0 for i in range(N-1)])
        
    elif mode == 'NMF':
        # B = np.array([[1.0, 0.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 0.0],
        #               [0.0, 0.0, 1.0, 0.0],
        #              [0.0, 0.0, 0.0, 1.0]])
        B = np.eye(N)
    
    elif mode == 'NMTF':
        # B = np.array([[1.0, 0.0, 0.0, 0.0],
        #               [0.0, 1.0, 0.0, 0.0],
        #               [0.0, 0.0, 1.0, 0.0],
        #              [0.0, 0.0, 0.0, 1.0]])
        B = np.eye(N)

    eps = 1e-8
    loss = []

    for _ in range (epochs):
        U = (U * (X @ M @ B.T)) / (U @ B @ M.T @ M @ B.T)
        if mode in ['BSMF', 'NMF', 'NMTF']:
            U -= U @ (lambda1 * np.ones((N, N)) + lambda2 * np.eye(N))
        U = np.clip(U, eps, np.inf)

        M = (M * (X.T @ U @ B)) / (M @ B.T @ U.T @ U @ B)

        if mode in ['BSMF', 'NMF', 'NMTF']:
            M -= M @ (lambda1 * np.ones((N, N)) + lambda2 * np.eye(N))
        
        M = np.clip(M, eps, np.inf)

        if mode == 'NMTF':
            B = (B * (U.T @ X @ M)) / (U.T @ U @ B @ M.T @ M)
            B = np.clip(B, eps, np.inf)
    
        iterLoss = lossFunc(X, U, B, M, lambda1, lambda2)
        loss.append(iterLoss)

        U = U / U.sum(axis=1).reshape(-1, 1)
        M = M / M.sum(axis=1).reshape(-1, 1)
        
    return U, M, loss, B


