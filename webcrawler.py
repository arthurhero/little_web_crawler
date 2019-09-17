import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os.path
import numpy as np
import math

lemmatizer = WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

url_frontier=list() #(url,parent_id)
url_crawled=list()
index=dict() #index[term]=list(doc_ids)
docs=list() #docs[doc_id]=doc_url
web_graph=list() #web_graph[doc_id]=([parent_ids],[child_ids])

# number of appearances for words per document 
word_frequency=list() #word_frequency[doc_id]=dict(term:frequency)
# number of documents per word
total_word_freq=dict() #total_word_freq[term]=frequency
pageranks=list() #pagerank[doc_id]=ranking

seed_url='https://en.wikipedia.org/wiki/Socrates'
crawl_num=300

#file names
index_fname='index.txt'
docs_fname='docs.txt'
word_freq_fname='wf.txt'
total_word_freq_fname='twf.txt'
pagerank_fname='pr.txt'

def url_filter(links, doc_id):
    '''
    take a list of links
    check whether to add this url to frontier or not
    return valid links
    '''
    valid_links=[]
    for link in links:
        if link in url_crawled:
            web_graph[docs.index(link)][0].append(doc_id)
            web_graph[doc_id][1].append(docs.index(link))
        elif "/wiki/" == link[:6] and link not in valid_links\
                and "File" not in link and "Special" not in link\
                and ":" not in link:
            valid_links.append("https://en.wikipedia.org"+link)
    return valid_links

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
    if url in url_crawled:
        return
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
        if getting_link:
            if cur == "''":
                # reaches the end of url
                getting_link=False
                if len(cur_link)>1:
                    links.append(cur_link)
                cur_link=''
            else:
                cur_link+=cur
            i+=1
            continue
        if cur == '<' and not inside_tag:
            inside_tag=True
            i+=1
            continue
        if cur == '>' and inside_tag:
            inside_tag=False
            i+=1
            continue
        if cur == "href=" and inside_tag:
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

    #add this page to the web graph
    web_graph.append((list(),list()))
    if parent_id>=0:
        web_graph[parent_id][1].append(doc_id)
        web_graph[doc_id][0].append(parent_id)
    #process the links and add to the frontier
    filtered_links=url_filter(links,doc_id)
    link_pairs=[(l,doc_id) for l in filtered_links]
    url_frontier.extend(link_pairs[:2])
    #add to docs
    docs.append(url)
    url_crawled.append(url)

def pagerank():
    '''
    takes in the web_graph and construct the pagerank list
    '''
    global pageranks
    epsilon = 10e-9
    alpha = 0.85
    graph = [np.array(node) for node in web_graph]
    num_doc = len(docs)
    initial_rank = 1/num_doc
    constant_factor = (1-alpha)/num_doc
    cur_rank = np.repeat(initial_rank, num_doc)
    prev_rank = [0]*len(cur_rank)
    while (sum(abs(cur_rank-prev_rank)) > epsilon):
        prev_rank = cur_rank.copy()
        for i in range(num_doc):
            # web_graph is a list of (parents, children)
            if len(graph[i][0])==0:
                cur_rank[i] = constant_factor
            else:
                parent_scores=cur_rank[graph[i][0]]
                parent_child_num=[len(graph[parent][1]) for parent in graph[i][0]]
                cur_rank[i] = constant_factor+alpha*np.sum(parent_scores/parent_child_num)
        cur_rank=cur_rank/np.sum(cur_rank)
    pageranks = cur_rank

def freqrank(pages,words):
    '''
    takes in a list of doc_ids and a list of query words
    rank the pages according to the word frequency
    '''
    #term frequencies
    page_len = []
    for page in pages:
        page_len.append(sum(word_frequency[page].values()))
    term_freq = []
    for page in pages:
        word_freq= []
        for word in words:
            word_freq.append(word_frequency[page][word])
        term_freq.append(sum(word_freq))
    #inverted document frequencies
    idf = math.log(len(docs)/len(pages))
    return np.array(term_freq)*idf
    #return np.array(term_freq)/np.array(page_len)

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
    f=open(index_fname,"w")
    for w,l in index.items():
        f.write(w)
        f.write(' ')
        for i in l:
            f.write(str(i))
            f.write(' ')
        f.write('\n')
    f.close()

    f=open(docs_fname,"w")
    for d in docs:
        f.write(d)
        f.write('\n')
    f.close()

    f=open(word_freq_fname,"w")
    for i in range(len(word_frequency)):
        f.write(str(i))
        f.write('\n')
        for w,fq in word_frequency[i].items():
            f.write(w)
            f.write(' ')
            f.write(str(fq))
            f.write('\n')
        f.write('\n')
    f.close()

    f=open(total_word_freq_fname,"w")
    for w,fq in total_word_freq.items():
        f.write(w)
        f.write(' ')
        f.write(str(fq))
        f.write('\n')
    f.close()

    f=open(pagerank_fname,"w")
    for p in pageranks:
        f.write(str(p))
        f.write('\n')
    f.close()

def parse_files():
    '''
    parse the five files and store them into the global variables
    '''
    f=open(index_fname,"r")
    lines=f.read().splitlines()
    for l in lines:
        toks=nltk.word_tokenize(l)
        w=toks.pop(0)
        toks=[int(i) for i in toks]
        index[w]=toks
    f.close()

    f=open(docs_fname,"r")
    global docs
    docs=f.read().splitlines()
    f.close()

    f=open(word_freq_fname,"r")
    lines=f.read().splitlines()
    newdoc=True
    for l in lines:
        if newdoc:
            word_frequency.append(dict())
            newdoc=False
        else:
            toks=nltk.word_tokenize(l)
            if len(toks)==2:
                w=toks[0]
                fq=int(toks[1])
                word_frequency[-1][w]=fq
            else:
                newdoc=True
    f.close()

    f=open(total_word_freq_fname,"r")
    lines=f.read().splitlines()
    for l in lines:
        toks=nltk.word_tokenize(l)
        w=toks[0]
        fq=int(toks[1])
        total_word_freq[w]=fq
    f.close()

    f=open(pagerank_fname,"r")
    lines=f.read().splitlines()
    global pageranks
    pageranks=[float(s) for s in lines]
    f.close()
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
        if w not in words:
            words.append(w)
    # look for documents starting from the most infrequent word
    results=list()
    # get the total frequency for the words
    freqs = list()
    for i in range(len(words)):
        if w not in total_word_freq:
            return []
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
                if cd in cur_results:
                    results.append(cd)
    if len(results)==0:
        return []
    # rank the documents according to page rank
    #print(len(pageranks))
    score1=[pageranks[i] for i in results]
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
            os.path.isfile(total_word_freq_fname) and\
            os.path.isfile(pagerank_fname):
        print('parsing files!...')
        parse_files()
        print('finished parsing!')
    else:
        print('start crawling!...')
        crawl()
        #print(web_graph)
        print('finished crawling!')
        '''
        print(docs)
        print(web_graph)
        print(pageranks)
        print(total_word_freq)
        print(word_frequency)
        print(index)
        '''

    print('')
    while True:
        query = input("please enter your query (enter q to quit): ") 
        if query == "q":
            break
        print('start searching!...')
        results = search(query) # a list of doc_ids
        print('finished searching! Here are the results:')
        # print out the resulting url
        for r in results:
            print(docs[r])
        print('')
