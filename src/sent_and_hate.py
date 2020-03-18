import pandas as pd
import csv
import nltk
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_hate_lexicon():
    """
    Loads all the unigrams, bigrams, and trigrams in our hate lexicon.
    """
    lexicon_path = '../hate-lexicon/'

    a1 = []
    with open(lexicon_path+'wydl.txt', 'r') as f:
        lines = f.read().splitlines()
    for row in lines:
        word = row.split(':')[0]
        a1.append(word.replace('"', '').rstrip())

    a2 = []
    temp = pd.read_csv(lexicon_path+'hatebase_dict.csv', header=None).iloc[:,0].tolist()
    for w in temp:
        a2.append(w.replace("'", '').replace(",", ''))

    a3 = pd.read_csv(lexicon_path+'refined_ngram_dict.csv', header=None).iloc[:,0].tolist()

    with open(lexicon_path+'badwords.txt', 'r') as f:
        a4 = f.read().splitlines()

    temp = a1 + a2 + a3 + a4
    hate_lexicon = set([x.lower() for x in temp])
    print("There are {} words/n-grams in our hate lexicon.".format(len(hate_lexicon)))
    return hate_lexicon


def get_hate_word_count(comment):
    """
    Counts the number of hate words/bigrams/trigrams (from our lexicon) present in the comment.

    Parameters
    ----------
    comment: str

    Returns
    -------
    int
        Count of the number of hate words
    """
    words = nltk.word_tokenize(comment)
    count = 0
    for n in [1,2,3]:
        for tup in ngrams(words, n):
            temp = '' # temp is the unigram/bigram/trigram
            for word in tup:
                temp += ' ' + word.lower()
            temp = temp.strip()
            if temp in HATE_LEXICON: # check if it exists in lexicon
                count += 1
    return count


def add_hate_col():
    """
    Add hate-count column to DataFrame.
    """
    counter = 0
    map_ID_hatecount = {}
    for ID, text in map_ID_text.items():
        try:
            map_ID_hatecount[ID] = get_hate_word_count(text)
        except:
            print("HateCountError for: ", ID, text)
        
        counter += 1
        if counter % 1000000 == 0:
            print("Done with: ", counter)

    # Add columns to DataFrame:
    df['hate_count'] = df['id'].map(map_ID_hatecount)


def add_sentiment_cols():
    """
    Add two sentiment columns to the DataFrame.
    """
    analyzer = SentimentIntensityAnalyzer()
    map_ID_compound = {} # compound score is the final sentiment score
    map_ID_neg = {} # negative score is a ratio (does not take word-order into account)

    counter = 0 # to keep track
    skipped_ids = [] # IDs skipped because text is empty

    for ID, text in map_ID_text.items():
        try:
            senti = analyzer.polarity_scores(text)
            map_ID_compound[ID] = senti['compound']
            map_ID_neg[ID] = senti['neg']
        except:
            skipped_ids.append(ID)

        counter += 1
        if counter % 1000000 == 0:
            print("Done with: ", counter)

    print("Total: ", len(map_ID_compound), len(map_ID_neg), "\nSkipped: ", len(skipped_ids))

    # Add columns to DataFrame:
    df['compound_VADER'] = df['id'].map(map_ID_compound)
    df['neg_VADER'] = df['id'].map(map_ID_neg)

    # Write Skipped IDs to a CSV:
    with open('../data/skipped_ids.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(skipped_ids)

        
if __name__ == '__main__':
    df = pd.read_csv('../data/dataset_raw.tsv', delimiter='\t')
    print("DataFrame size: ", df.shape)

    map_ID_text = dict((tuple(x) for x in df[['id', 'text']].values))
    print("ID-text dictionary: ", len(map_ID_text))

    HATE_LEXICON = load_hate_lexicon()
    print("Add hate-count column:")
    add_hate_col()
    print("\n\nAdd sentiment columns:")
    add_sentiment_cols()

    df.to_csv('../data/dataset_with_sent_hate.tsv', sep='\t', index=None)