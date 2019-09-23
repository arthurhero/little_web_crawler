'''
CSC 395 Assignment 1
'''
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os.path
import numpy as np
import math

# initialize lemmatizer and get stop word list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# initialize data structures for storing calculated data
url_frontier = list()  # a queue of to-be-parsed urls
                       # element in this list has the form (url, parent_id)
url_crawled = list()
index = dict()  # a dictionary for inverted index: index[term]=list(doc_ids)
docs = list()  # a list of all urls docs[doc_id]=doc_url
web_graph = list()  # graph of links: web_graph[doc_id]=([parent_ids],[child_ids])

word_frequency = list()  # term frequencies for each document
                         # word_frequency[doc_id]=dict(term:frequency)
total_word_freq = dict()  # number of documents per term
                          # total_word_freq[term] = frequency
pageranks = list()  # pagerank[doc_id]=ranking

seed_url = 'https://en.wikipedia.org/wiki/Philosophy'  # this is our 'starting point' url
crawl_num = 100  # upper limit for number of crawled pages

# names for files that store precomputed statistics and extracted info
index_fname = 'index.txt'
docs_fname = 'docs.txt'
word_freq_fname = 'wf.txt'
total_word_freq_fname = 'twf.txt'
pagerank_fname = 'pr.txt'


def url_filter(links, doc_id):
    '''
    take a list of links
    check whether to add this url to frontier or not

    return valid links
    '''
    valid_links = []
    for link in links:
        if link in url_crawled:
            # update this link's parent list
            web_graph[docs.index(link)][0].append(doc_id)
            # update doc_id's child list
            web_graph[doc_id][1].append(docs.index(link))
        # if link is a valid wiki link that's not already in the list of valid links
        # we add it to the list of valid links
        elif "/wiki/" == link[:6] and link not in valid_links\
                and "File" not in link and "Special" not in link\
                and "disambiguation" not in link\
                and ":" not in link:
            valid_links.append("https://en.wikipedia.org"+link)
    return valid_links


def process_words(w):
    '''
    takes a list of words
    return a dict of word: frequency
    '''
    w.sort()  # sort the list so that same words are grouped together
    # construct the dict by going over all words in the list w
    word_count = dict()
    last = ''
    for i in range(len(w)):
        if w[i] == last:
            # if same as last word, increment count
            word_count[w[i]] += 1
        else:
            # if a new word, set count to 1 and set this word as the last-seen one
            word_count[w[i]] = 1
            last = w[i]
    return word_count


def parse_page(url, parent_id):
    '''
    parse one page
    adds this page to docs
    adds the links on this page to the frontier
    add this page to the web_graph list
    incorporate the word frequency of this page
    add url to url_crawled
    '''
    if url in url_crawled:
        return
    # request to open the url
    try:
        page = urllib.request.urlopen(url)
    except ValueError:
        return
    except urllib.error.URLError:
        return
    except ConnectionResetError:
        return
    except TimeoutError:
        return

    # raw contents read from the page
    page_raw = str(page.read())

    # initialize data structures for storing words and links in the page
    words = list()
    links = list()

    # whether we are inside an html tag or not
    inside_tag = False

    # parse into tokens
    tokens_raw = nltk.word_tokenize(page_raw)

    # temp storage for url
    cur_link = ''
    # whether we are getting a url or not
    getting_link = False
    # position in token list
    i = 0
    # go over the tokens
    while i < len(tokens_raw):
        cur = tokens_raw[i]  # current token
        if getting_link:  # if reading a url
            if cur == "''":  # if reaches the end of url
                getting_link = False
                if len(cur_link) > 1:  # if a possibly valid link
                    links.append(cur_link)
                cur_link = ''  # reset cur_link
            else:  # if haven't reached the end of url, append to the link string
                cur_link += cur
            i += 1
            continue
        # if encountering '<', which starts a tag of html
        if cur == '<' and not inside_tag:
            inside_tag = True
            i += 1
            continue
        if cur == '>' and inside_tag:  # if reaches the end of an html tag
            inside_tag = False
            i += 1
            continue
        if cur == "href=" and inside_tag:  # if reaches 'href=', an html attribute for links
            getting_link = True
            i += 2
            continue
        if not inside_tag and cur.isalpha() and len(cur) > 1:  # if reaches a valid word
            word = lemmatizer.lemmatize(cur.lower())  # lemmatize the lower-cased word
            if word not in stop_words:
                words.append(word)  # append to list of words if not a stop word
        i += 1
        continue

    doc_id = len(docs) # id for this current being parsed link

    # get a dict of word count
    word_count = process_words(words)
    # append word counts for this page
    word_frequency.append(word_count)
    # update inverted index and total_word_freq
    for w in word_count:
        # check whether this word had been seen before
        if w in index.keys():
            # add this id for this page to the posting list for w
            index[w].append(doc_id)
        else:
            # create a posting list for w
            index[w] = [doc_id]

        if w in total_word_freq.keys():
            # increment number of occurrences for w
            total_word_freq[w] += 1
        else:
            # create an entry for w, setting number of occurrences to 1
            total_word_freq[w] = 1

    # add this page to the web graph
    web_graph.append((list(),list()))
    if parent_id >= 0:
        # include this page into its parent's child list
        web_graph[parent_id][1].append(doc_id)
        # add this page's parent to the parent list
        web_graph[doc_id][0].append(parent_id)
    # process the links and add to the frontier
    filtered_links = url_filter(links, doc_id)
    link_pairs = [(l, doc_id) for l in filtered_links]
    url_frontier.extend(link_pairs[:2])
    # add to list of docs and list of crawled urls
    docs.append(url)
    url_crawled.append(url)


def pagerank():
    '''
    takes in the web_graph and construct the pagerank list
    '''
    global pageranks
    # pagerank algorithm parameters
    epsilon = 10e-5
    alpha = 0.85
    num_doc = len(docs)
    initial_rank = 1/num_doc
    # constant part of the page rank formula
    constant_factor = (1-alpha)/num_doc
    # initialize current ranks and previous ranks
    cur_rank = np.repeat(initial_rank, num_doc)
    prev_rank = [0]*num_doc
    # while the rankings have not converged (stabled)
    while sum(abs(cur_rank-prev_rank)) > epsilon:
        prev_rank = cur_rank.copy()
        for i in range(num_doc):
            # if this doc has no parent
            if len(web_graph[i][0]) == 0:
                cur_rank[i] = constant_factor
            else:
                # if this doc has parent(s), retrieve rank(s) of parent(s)
                parent_ranks = cur_rank[web_graph[i][0]]
                # get numbers of children for each parent
                parent_child_num = [len(web_graph[parent][1]) for parent in web_graph[i][0]]
                # update rank according to pagerank formula
                cur_rank[i] = constant_factor+alpha*np.sum(parent_ranks/parent_child_num)
        # normalize ranks
        cur_rank=cur_rank/np.sum(cur_rank)
    # store cur_rank to the global variable pageranks
    pageranks = cur_rank


def freqrank(pages, words):
    '''
    takes in a list of doc_ids and a list of query words
    each doc in this list of doc_ids has all of query words in it
    return tf-idf for each doc
    '''
    num_page = len(pages)
    num_term = len(words)
    # term frequency matrix; dimensions: num_page x num_term
    tf = np.zeros((num_page, num_term))
    # inverse document frequency list; length will be num_term
    idf = list()
    # compute tf for each (doc, term) pair
    for i in range(num_page):
        for j in range(num_term):
            tf[i][j] = 1+math.log(word_frequency[pages[i]][words[j]])
    # compute idf for each term
    for word in words:
        idf.append(math.log(1+num_page/total_word_freq[word]))
    # calculate and return tf-idf for each doc
    return np.sum(np.multiply(tf, idf), axis=1)


def crawl():
    '''
    start to crawl from the seed_url up to crawl_num pages
    '''
    # add seed_url to url_frontier
    # since we start from seed_url, we assume it has no parent
    # and thus set its parent's id to -1
    url_frontier.append((seed_url, -1))
    # while we haven't crawled crawl_num pages
    # and there is an url waiting for processing in frontier
    while len(docs) < crawl_num and len(url_frontier) > 0:
        # parse the first page in the frontier
        cur_url, cur_par = url_frontier.pop(0)
        parse_page(cur_url, cur_par)
    # calculate page ranks for the pages
    pagerank()
    # write data to disk so we don't have to recalculate next time we run
    write_files()


def write_files():
    '''
    write the five global data structures to disk
    '''
    # write inverted index to disk
    f = open(index_fname, "w")
    for w, l in index.items():
        f.write(w)
        f.write(' ')
        for i in l:
            f.write(str(i))
            f.write(' ')
        f.write('\n')
    f.close()
    # write list of docs to disk
    f = open(docs_fname, "w")
    for d in docs:
        f.write(d)
        f.write('\n')
    f.close()
    # write word frequencies of each doc to disk
    f = open(word_freq_fname, "w")
    for i in range(len(word_frequency)):
        f.write(str(i))
        f.write('\n')
        for w, fq in word_frequency[i].items():
            f.write(w)
            f.write(' ')
            f.write(str(fq))
            f.write('\n')
        f.write('\n')
    f.close()
    # write total word frequencies to disk
    f = open(total_word_freq_fname, "w")
    for w, fq in total_word_freq.items():
        f.write(w)
        f.write(' ')
        f.write(str(fq))
        f.write('\n')
    f.close()
    # write page rank to disk
    f = open(pagerank_fname, "w")
    for p in pageranks:
        f.write(str(p))
        f.write('\n')
    f.close()


def parse_files():
    '''
    parse the five files and store them into the global variables
    '''
    # set up inverted index
    f = open(index_fname, "r")
    global index
    lines = f.read().splitlines()
    for l in lines:
        toks = nltk.word_tokenize(l)
        w = toks.pop(0)
        toks = [int(i) for i in toks]
        index[w] = toks
    f.close()
    # set up docs we have
    f = open(docs_fname, "r")
    global docs
    docs = f.read().splitlines()
    f.close()
    # set up word frequency dictionaries for each doc
    f = open(word_freq_fname, "r")
    global word_frequency
    lines = f.read().splitlines()
    newdoc = True
    for l in lines:
        if newdoc:
            word_frequency.append(dict())
            newdoc = False
        else:
            toks = nltk.word_tokenize(l)
            if len(toks) == 2:
                w = toks[0]
                fq = int(toks[1])
                word_frequency[-1][w] = fq
            else:
                newdoc = True
    f.close()
    # set up total word frequencies
    f = open(total_word_freq_fname, "r")
    global total_word_freq
    lines = f.read().splitlines()
    for l in lines:
        toks = nltk.word_tokenize(l)
        w = toks[0]
        fq = int(toks[1])
        total_word_freq[w] = fq
    f.close()
    # set up page ranks
    f = open(pagerank_fname, "r")
    global pageranks
    lines = f.read().splitlines()
    pageranks = [float(s) for s in lines]
    f.close()
    return


def search(query):
    '''
    search all docs containing the words
    rank the result based on pagerank and tf-idf
    return a list of doc_ids as search result
    '''
    # parse the query into words
    words = list()
    qtokens = nltk.word_tokenize(query)
    # standardize query words
    for qt in qtokens:
        w = lemmatizer.lemmatize(qt.lower())
        # create set of query terms
        if w not in words:
            words.append(w)
    # look for documents that contain all query terms
    # by starting from the most infrequent word
    results = list()
    freqs = list()
    # get the total frequency for the words
    for w in words:
        # if not having any document containing w
        # then there's nothing in search result
        if w not in total_word_freq:
            return []
        freqs.append(total_word_freq[w])
    freqs = np.asarray(freqs)
    # indices of sorted elements of freqs (from lowest to highest)
    freq_rank = np.argsort(freqs)
    # do boolean retrieval, starting with the most infrequent word
    for i in range(len(words)):
        ind = freq_rank[i]
        w = words[ind]  # the most infrequent word
        cur_docs = index[w]  # posting list for w
        if len(results) == 0:
            results.extend(cur_docs)
        else:
            cur_results = results
            results = list()
            # intersect posting list for w with previous results
            # to get documents that contain all query words
            for cd in cur_docs:
                if cd in cur_results:
                    results.append(cd)
    # if there is no document that contains all query words, we return nothing
    if len(results) == 0:
        return []
    # rank the documents according to page rank
    score1 = [pageranks[i] for i in results]
    score1 = np.asarray(score1)
    score1 = score1/np.sum(score1)
    # rank the documents according to tf-idf
    score2 = freqrank(results, words)
    score2 = np.asarray(score2)
    score2 = score2/np.sum(score2)
    # combine the two rankings
    print("pagerank:", score1)
    print("freqrank:", score2)
    score = score1+score2
    # indices of sorted elements of score (from lowest to highest)
    ranking = np.argsort(score)
    ranking = list(ranking)
    # indices of sorted elements of score (from highest to lowest)
    ranking.reverse()
    # final result that contains doc_ids sorted by their scores
    # from highest to lowest
    final = list()
    for i in range(len(ranking)):
        final.append(results[ranking[i]])
    return final


if __name__== "__main__":
    # check whether need to crawl by checking if the five files
    # for storing data structures exist or not
    if os.path.isfile(index_fname) and\
            os.path.isfile(docs_fname) and\
            os.path.isfile(word_freq_fname) and\
            os.path.isfile(total_word_freq_fname) and\
            os.path.isfile(pagerank_fname):
        print('parsing files!...')
        parse_files()
        print('finished parsing!')
    # if we don't have local files for calculated data
    # start by crawling and then calculating data
    else:
        print('start crawling!...')
        crawl()
        print('finished crawling!')
    print('')
    # ask for user input
    while True:
        query = input("please enter your query (enter q to quit): ")
        # if user asks to quit
        if query == "q":
            break
        # else, the user has entered valid input query
        print('start searching!...')
        results = search(query) # a list of doc_ids
        print('finished searching! Here are the results:')
        # print out the resulting urls
        for r in results:
            print(docs[r])
        print('')