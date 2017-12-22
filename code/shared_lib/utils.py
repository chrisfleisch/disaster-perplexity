import re
import time
import itertools
import numpy as np

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML
import jinja2





from subprocess import call
from datetime import datetime
import string
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import glob
from collections import Counter
import vocabulary
import ngram_lm
import ngram_utils
import simple_trigram

def flatten(list_of_lists):
    """Flatten a list-of-lists into a single list."""
    return list(itertools.chain.from_iterable(list_of_lists))

HIGHLIGHT_BUTTON_TMPL = jinja2.Template("""
<script>
colors_on = true;
function color_cells() {
  var ffunc = function(i,e) {return e.innerText {{ filter_cond }}; }
  var cells = $('table.dataframe').children('tbody')
                                  .children('tr')
                                  .children('td')
                                  .filter(ffunc);
  if (colors_on) {
    cells.css('background', 'white');
  } else {
    cells.css('background', '{{ highlight_color }}');
  }
  colors_on = !colors_on;
}
$( document ).ready(color_cells);
</script>
<form action="javascript:color_cells()">
<input type="submit" value="Toggle highlighting (val {{ filter_cond }})"></form>
""")

RESIZE_CELLS_TMPL = jinja2.Template("""
<script>
var df = $('table.dataframe');
var cells = df.children('tbody').children('tr')
                                .children('td');
cells.css("width", "{{ w }}px").css("height", "{{ h }}px");
</script>
""")

def render_matrix(M, rows=None, cols=None, dtype=float,
                        min_size=30, highlight=""):
    html = [pd.DataFrame(M, index=rows, columns=cols,
                         dtype=dtype)._repr_html_()]
    if min_size > 0:
        html.append(RESIZE_CELLS_TMPL.render(w=min_size, h=min_size))

    if highlight:
        html.append(HIGHLIGHT_BUTTON_TMPL.render(filter_cond=highlight,
                                             highlight_color="yellow"))

    return "\n".join(html)
    
def pretty_print_matrix(*args, **kwargs):
    """Pretty-print a matrix using Pandas.

    Optionally supports a highlight button, which is a very, very experimental
    piece of messy JavaScript. It seems to work for demonstration purposes.

    Args:
      M : 2D numpy array
      rows : list of row labels
      cols : list of column labels
      dtype : data type (float or int)
      min_size : minimum cell size, in pixels
      highlight (string): if non-empty, interpreted as a predicate on cell
      values, and will render a "Toggle highlighting" button.
    """
    html = render_matrix(*args, **kwargs)
    display(HTML(html))


def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


##
# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

##
# Data loading functions
import nltk
import vocabulary

def get_corpus(name="brown"):
    return nltk.corpus.__getattr__(name)

def sents_to_tokens(sents, vocab):
    """Returns an flattened list of the words in the sentences, with normal padding."""
    padded_sentences = (["<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([canonicalize_word(w, wordset=vocab.wordset)
                     for w in flatten(padded_sentences)], dtype=object)

def build_vocab(corpus, V=10000):
    token_feed = (canonicalize_word(w) for w in corpus.words())
    vocab = vocabulary.Vocabulary(token_feed, size=V)
    return vocab

def get_train_test_sents(corpus, split=0.8, shuffle=True):
    """Get train and test sentences.

    Args:
      corpus: nltk.corpus that supports sents() function
      split (double): fraction to use as training set
      shuffle (int or bool): seed for shuffle of input data, or False to just
      take the training data as the first xx% contiguously.

    Returns:
      train_sentences, test_sentences ( list(list(string)) ): the train and test
      splits
    """
    sentences = np.array(corpus.sents(), dtype=object)
    fmt = (len(sentences), sum(map(len, sentences)))
    print "Loaded %d sentences (%g tokens)" % fmt

    if shuffle:
        rng = np.random.RandomState(shuffle)
        rng.shuffle(sentences)  # in-place
    train_frac = 0.8
    split_idx = int(train_frac * len(sentences))
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]

    fmt = (len(train_sentences), sum(map(len, train_sentences)))
    print "Training set: %d sentences (%d tokens)" % fmt
    fmt = (len(test_sentences), sum(map(len, test_sentences)))
    print "Test set: %d sentences (%d tokens)" % fmt

    return train_sentences, test_sentences

def preprocess_sentences(sentences, vocab):
    """Preprocess sentences by canonicalizing and mapping to ids.

    Args:
      sentences ( list(list(string)) ): input sentences
      vocab: Vocabulary object, already initialized

    Returns:
      ids ( array(int) ): flattened array of sentences, including boundary <s>
      tokens.
    """
    # Add sentence boundaries, canonicalize, and handle unknowns
    words = flatten(["<s>"] + s + ["</s>"] for s in sentences)
    words = [canonicalize_word(w, wordset=vocab.word_to_id)
             for w in words]
    return np.array(vocab.words_to_ids(words))

##
# Use this function
def load_corpus(name, split=0.8, V=10000, shuffle=0):
    """Load a named corpus and split train/test along sentences."""
    corpus = get_corpus(name)
    vocab = build_vocab(corpus, V)
    train_sentences, test_sentences = get_train_test_sents(corpus, split, shuffle)
    train_ids = preprocess_sentences(train_sentences, vocab)
    test_ids = preprocess_sentences(test_sentences, vocab)
    return vocab, train_ids, test_ids

##
# Use this function
def batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) / batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]

    
    
    
    

def download_tweets(month_prefix, location, begin_day, end_day, max_tweets_per_day=2000):
    for i in xrange(begin_day, end_day):
        frmday = str(i)
        if i < 10:
            frmday = "0" + frmday
        today = str(i+1)
        if i+1 < 10:
            today = "0" + today
        frm = month_prefix+frmday
        until = month_prefix+today
        output_dir = 'data_'+location.replace(" ", "_")+ '/'
        call(['mkdir', output_dir])
        cmd = ["python", "GetOldTweets-python/Exporter.py", "--since", frm, "--until", until, "--near", location, "--maxtweets", str(max_tweets_per_day), "--output", "data/" + output_dir + frm +".csv"]
        call(cmd)
        
def get_train_and_tests(directory, train_size=21, test_size=3):
    allFiles = sorted(glob.glob(directory + "/*.csv"))
    train_frame = pd.DataFrame()
    train_list = []
    tests_list = []
    test_list = []
    tweet_column = 'text'
    test_files = []
    tests_files = []
    for index, file_ in enumerate(allFiles):
        df = pd.read_csv(file_,index_col=None, header=0, delimiter=';', error_bad_lines=False)
        if index < train_size:
            #print "Adding {0} to training with {1} lines.".format(file_, len(df))
            train_list.append(df[tweet_column])
        elif (index + 1 - train_size) % test_size == 0:
            #print "Adding {0} to testing with {1} lines.".format(file_, len(df))
            test_list.append(df[tweet_column])
            test_files.append(file_)
            test_set = pd.concat(test_list, ignore_index=True).dropna()
            #print "Adding a test set to training with {0} lines.".format(len(test_set))
            tests_list.append(test_set)
            tests_files.append(test_files)
            test_list=[]
            test_files=[]
        else:
            #print "Adding {0} to testing with {1} lines.".format(file_, len(df))
            test_list.append(df[tweet_column])
            test_files.append(file_)

    train_frame = pd.concat(train_list, ignore_index=True).dropna()
    return (train_frame, tests_list, tests_files)

#Sanitizes the text by removing front and end punctuation, 
#making words lower case, and removing any empty strings.
def get_text_sanitized(tweet):
    return ' '.join([w.lower().strip().rstrip(string.punctuation)\
        .lstrip(string.punctuation).strip()\
        for w in tweet.replace('\xe2\x80\xa6', '').split(" ")\
        if w.strip().rstrip(string.punctuation).strip()])

#Gets the text, clean it, make it lower case, stem the words, and split
#into a vector. Also, remove stop words.
def get_text_normalized(tweet):
    #Sanitize the text first.
    text = get_text_sanitized(tweet).split()
    
    #Remove the stop words.
    text = [t for t in text if t not in [stopwords.words('english')] ]

    return text
    
    #Stemmer gets upset at a lot of tweets
    #Create the stemmer.
    stemmer = LancasterStemmer()
    
    #Stem the words.
    return [stemmer.stem(t) for t in text]

def purge_urls(tweet):
    return re.sub('http[s]?://(www. )?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)

def sanitize_dataset(dataset):
    sentences = []
    for sentence in dataset:
        sentences.append(get_text_normalized(purge_urls(sentence)))
    return sentences

def get_vocab_size(sentences):
    results = Counter()
    for sent in sentences:
        results.update(sent)
    return len(results)

def sents_to_tokens(vocab, sents):
    """Returns an flattened list of the words in the sentences, with padding for a trigram model."""
    padded_sentences = (["<s>", "<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([canonicalize_word(w, wordset=vocab.wordset) 
                     for w in flatten(padded_sentences)], dtype=object)

def build_model(train_tokens, addk=False, k=0.001, delta=0.75):
    # Uncomment the line below for the model you want to run.
    Model = ngram_lm.KNTrigramLM
    if addk:
        Model = ngram_lm.AddKTrigramLM
    t0 = time.time()
    print "Building trigram LM...",
    lm = Model(train_tokens)
    print "done in %.02f s" % (time.time() - t0)
    ngram_utils.print_stats(lm)
    lm.set_live_params(k = 0.001, delta=0.75)
    return lm

def get_perplexity(lm, tokens, train_or_test="Test"):
    #log_p_data, num_real_tokens = ngram_utils.score_seq(lm, train_tokens)
    #print "Train perplexity: %.02f" % (2**(-1*log_p_data/num_real_tokens))
    log_p_data, num_real_tokens = ngram_utils.score_seq(lm, tokens)
    perplexity = -1.0
    if num_real_tokens > 0:
        perplexity = (2**(-1*log_p_data/num_real_tokens))
    print "{0} perplexity: {1:.2f}".format(train_or_test, perplexity)
    return perplexity

def measure_perplexity_over_time(train_set, test_sets, test_files, addk=False):
    train_sents = sanitize_dataset(train_set)
    vocab = vocabulary.Vocabulary((canonicalize_word(w) for w in flatten(train_sents)), size=get_vocab_size(train_sents), unknown_size=500)
    print "Train set vocabulary: %d words" % vocab.size
    train_tokens = sents_to_tokens(vocab, train_sents)
    lm = build_model(train_tokens, addk)
    get_perplexity(lm, train_tokens, "Train")
    perplexities = []
    for idx, test_set in enumerate(test_sets):
        print "="*20
        print "Test set #{0} containing files:\n{1}".format(idx, test_files[idx])
        test_sents = sanitize_dataset(test_set)
        test_tokens = sents_to_tokens(vocab, test_sents)
        perplexities.append(get_perplexity(lm, test_tokens))
        print "="*20
    return perplexities
