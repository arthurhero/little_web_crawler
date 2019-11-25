# To Start...
1. Find the variable "seed_url" in the code file "webcrawler.py", and set this variable to an url of any wikipedia page.
   (Our search engine specifically focuses on wikipedia pages by having an url filter that filters out non-wiki pages.)
   Think of this url as the starting point of our search engine - this url will be the first page that our search engine crawls.
2. Find the variable "crawl_num" (which should be immediately below "seed_url" in the code).
   Set it to a number of desired. This number will be the upper limit for the number of crawled pages.
3. Run the code file. If you are running it for the first time, wait for the search engine crawling web pages. Then you will be prompted to enter your query.
   If you are running it not for the first time, wait for the search engine parsing existing data files (files for recorded documents, indices, etc.). Then you will be prompted to enter your query.

# Notes:
1. Query words are assumed to be connected with "AND" when a user searches using our engine.
   Thus, our engine will only look at documents that contain all the query words.
2. Our search engine does not autocorrect misspelled words. It searches for the exact words as entered by the user.
   Thus, please check your spelling when searching using our engine.
