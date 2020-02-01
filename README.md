# Narrative Detection Source Code (python3.7)

### A small demo
Check `Synthetic_data_code.ipynb`: a demo jupyter notebook on synthetic dataset (Hawaii Earthquake)!


### Content

- `/src/*`:  source code
- `/sample/*`:  a real-world twitter dataset


### Run synthetic dataset
Before running, please ```mkdir label``` first on the same folder.

```text
python run.py
```

### Run sample dataset
```text
python run.py --pathA sample/friend.txt --pathD sample/data.csv --pathK N --fastmode N --N 3 --kthreshold 10 --uthreshold 8
```

### Parameters
```text
$ python run.py -h
usage: run.py [-h] [--pathA PATHA] [--pathD PATHD] [--pathK PATHK]
              [--fastmode FASTMODE] [--seed SEED] [--l1 L1] [--l2 L2] [--N N]
              [--epochs EPOCHS] [--process PROCESS] [--kthreshold KTHRESHOLD]
              [--uthreshold UTHRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --pathA PATHA         path of adjacency relation
  --pathD PATHD         path of main data file
  --pathK PATHK         path of keywordList
  --fastmode FASTMODE   use M-module or not (Y or N)
  --seed SEED           Random seed.
  --l1 L1               coefficient of l1-norm
  --l2 L2               coefficient of l2-norm
  --N N                 dimension of beliefs.
  --epochs EPOCHS       number of iterations
  --process PROCESS     number of available processes
  --kthreshold KTHRESHOLD
                        the number of minimun keywords contained in processed
                        text
  --uthreshold UTHRESHOLD
                        the mininum frequency of user occurance
```

### Data format
`pathA`, `pathD` and `pathK` are the paths for user adjacency files, user-text main files and keyword files. If `pathA` is not specified, then the program will run on synthetic "Hawaii Earthquake" dataset. To run on customized dataset, `pathA`, `pathD` are necessary, but `pathK` is highly recommended for domain-specific tailoring. The basic format of these three files is like:
```text
# pathA xxx.txt

tonyman7777 DailySabah
jimineuropa Ericamarat
ppsabengel  VeraVanHorne
ppsabengel  jamala
NinaSublattifan Eurovision
...
JustMeLuka  AmbassadorPower

# each row is a (user1, user2) pair separated by `\t`.


# pathD xxx.csv

name    rawTweet
tonyman7777 RT @DailySabah: Crimean Tatar Eurovision winner Jamala to get Ukrainian honor https://t.co/DW8eXhgTE8 https://t.co/xLvO4SMS5T
...
xLoveWavex  RT @jacob__017: Honestly congratulations Jamala &amp; Ukraine, you really deserved it ❤️
knight_of_sg    I added a video to a @YouTube playlist https://t.co/ol1PhFmmcW Jamala - 1944 (Ukraine) 2016 Eurovision Song Contest

# a DataFrame with two fields "name" and "rawTweet" defined by Pandas. The first row is the filed names, and the rest of the rows (user, tweet) pair seperated by `\t`.


# pathK xxx.txt

keyword1
keyword2
...
keyword100

# each row is a keyword
```

### Note
The software are totally based on "keyword similarity", "user dependency" and "user retweeting activity", so that the larger uthreshold (popular users) and kthreshold (long tweets) are, the better seperation it will give. `fastmode` is actually another way to compute message similarity, feel free to try.

