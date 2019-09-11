import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os.path
import numpy as np

lemmatizer = WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

url_frontier=list() #(url,parent_id)
url_crawled=list()
index=dict() #index[term]=list(doc_ids)
docs=list() #docs[doc_id]=doc_url
web_graph=list() #web_graph[doc_id]=list(child_id)

# number of appearances for words per document 
word_frequency=list() #word_frequency[doc_id]=dict(term:frequency)
# number of documents per word
total_word_freq=dict() #total_word_freq[term]=frequency
pagerank=list() #pagerank[doc_id]=ranking

seed_url=''
crawl_num=1000

#file names
index_fname=''
docs_fname=''
word_freq_fname=''
total_word_freq_fname=''
pagerank_fname=''

def url_filter(links):
    '''
    take a list of links
    check whether to add this url to frontier or not
    return valid links
    '''
    # check if valid url
    # check duplicate
    return

def process_words(w):
    '''
    takes a list of words
    return a dict of word frequency 
    '''
    w.sort()
    # construct the word frequency dict
    word_count=dict()
    last=''
    for i in range(len(w)):
        if w[i]==last:
            word_count[w[i]]+=1
        else:
            word_count[w[i]]=1
            last=w[i]
    return word_count

def parse_page(url,parent_id):
    '''
    parse one page
    adds that page to docs
    adds the links on this page to the frontier and 
    to the web_graph list
    adds the word frequency of this page to the list of word frequencies
    add url to url_crawled
    '''
    # request the web
    page = urllib.request.urlopen(url)
    #####need some error handling here
    page_raw = str(page.read())

    words=list()
    links=list()

    # whether we are inside an html tag or not
    inside_tag=False

    # parse into tokens
    tokens_raw = nltk.word_tokenize(page_raw)

    # tmp storage for url
    cur_link=''
    # whether we are getting a url or not
    getting_link=False

    i=0
    #start parsing the page
    while i < len(tokens_raw):
        cur=tokens_raw[i]
        #print("token:",cur)
        if getting_link:
            if cur == "''":
                # reaches the end of url
                getting_link=False
                if len(cur_link)>1:
                    links.append(cur_link)
                cur_link=''
            else:
                cur_link+=cur
                #print(cur_link)
            i+=1
            continue
        if cur == '<' and not inside_tag:
            #print("in!")
            inside_tag=True
            i+=1
            continue
        if cur == '>' and inside_tag:
            #print("out!")
            inside_tag=False
            i+=1
            continue
        if cur == "href=" and inside_tag:
            #print("href!")
            getting_link=True
            i+=2
            continue
        if not inside_tag and cur.isalpha() and len(cur)>1:
            word=lemmatizer.lemmatize(cur.lower()) # lemmatize
            if word not in stop_words:
                words.append(word)
        i+=1
        continue

    doc_id=len(docs)

    #get word count
    word_count=process_words(words)
    word_frequency.append(word_count)
    #add to index and total_word_freq
    for w in word_count:
        # check whether this word had been seen before
        if w in index.keys():
            index[w].append(doc_id)
        else:
            index[w]=[doc_id]
        if w in total_word_freq.keys():
            total_word_freq[w]+=1
        else:
            total_word_freq[w]=1

    #process the links and add to the frontier
    filtered_links=url_filter(links)
    link_pairs=[(l,doc_id) for l in filtered_links]
    url_frontier.extend(link_pairs)

    #add this page to the web graph
    if parent_id>=0:
        web_graph[parent_id].append(doc_id)
    web_graph.append(list())

    #add to docs
    docs.append(url)
    url_crawled.append(url)

def pagerank():
    '''
    takes in the web_graph and construct the pagerank list
    '''

def freqrank(pages,words):
    '''
    takes in a list of doc_ids and a list of query words
    rank the pages according to the word frequency
    '''

def crawl():
    '''
    start to crawl from the seed_url up to crawl_num pages
    '''
    url_frontier.append((seed_url,-1))
    while len(docs)<crawl_num and len(url_frontier)>0:
        # parse the first page in the frontier
        cur_url,cur_par=url_frontier.pop(0)
        parse_page(cur_url,cur_par)
    #pagerank the pages
    pagerank()
    #write the files to disk
    write_files()

def write_files():
    '''
    write the five files to disk
    '''

def parse_files():
    '''
    parse the five files and store them into the global variables
    '''
    return

def search(query):
    '''
    search all docs containing the words
    rank the result based on pagerank and word frequency
    return a list of doc_ids as search result
    '''
    # parse the query into words
    words=list()
    qtokens=nltk.word_tokenize(query)
    for qt in qtokens:
        w=lemmatizer.lemmatize(qt.lower())
        if w is not in words:
            words.append(w)
    # look for documents starting from the most infrequent word
    results=list()
    # get the total frequency for the words
    freqs = list()
    for i in range(len(words)):
        freqs.append(total_word_freq[w])
    freqs=np.asarray(freqs)
    freq_rank=np.argsort(freqs) # from lowest to highest
    candidates=list()
    for i in range(len(words)):
        ind=freq_rank[i]
        if freqs[ind]==0:
            continue
        w=words[ind]
        cur_docs=index[w]
        if len(results)==0:
            results.extend(cur_docs)
        else:
            cur_results=results
            results=list()
            for cd in cur_docs:
                if cd is in cur_results:
                    results.append(cd)
    if len(results)==0:
        return []
    # rank the documents according to page rank
    score1=[pagerank[i] for i in results]
    score1=np.asarray(score1)
    score1=score1/np.sum(score1)
    # rank the documents according to word frequency
    score2=freqrank(results,words)
    score2=np.asarray(score2)
    score2=score2/np.sum(score2)
    # combine the two rankings
    score=score1+score2
    ranking=np.argsort(score)
    ranking=list(ranking)
    ranking.reverse()
    # return the result
    final=list()
    for i in range(len(ranking)):
        final.append(results[ranking[i]])
    return final

if __name__== "__main__":
    # check whether need to crawl
    if os.path.isfile(index_fname) and\
            os.path.isfile(docs_fname) and\
            os.path.isfile(word_freq_fname) and\
            os.path.isfile(pagerank_fname):
        print('parsing files!...')
        parse_files()
        print('finished parsing!')
    else:
        print('start crawling!...')
        crawl()
        print('finished crawling!')

    print('')
    query = raw_input("please enter your query: ") 
    print('start searching!...')
    results = search(query) # a list of doc_ids
    print('finished searching! Here are the results:')
    # print out the resulting url
    for r in results:
        print(docs[r])
