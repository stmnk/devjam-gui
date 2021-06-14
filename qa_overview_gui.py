import os
import time
import json
import numpy as np  
import pandas as pd
import streamlit as st
import requests as req
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer


load_dotenv()
DETA_URL = os.getenv('DETA_URL')
BONSAI_URL = os.getenv('BONSAI_URL')
DETA_METHOD = 'GET'

# TODO: improve layout (columns, sidebar, forms)
# st.set_page_config(layout='wide')


st.title('DevJam: An overview of Question Answering and related NLP tasks')


##########################################################
st.subheader('1. A simple starting question')
##########################################################


WIKI_URL = 'https://en.wikipedia.org/w/api.php'
WIKI_QUERY = "?format=json&action=query&prop=extracts&explaintext=1"
WIKI_BERT = "&titles=BERT_(language_model)"
WIKI_METHOD = 'GET'

response = req.request(WIKI_METHOD, f'{WIKI_URL}{WIKI_QUERY}{WIKI_BERT}')
resp_json = json.loads(response.content.decode("utf-8"))
wiki_bert = resp_json['query']['pages']['62026514']['extract']
paragraph =  wiki_bert

written_passage = st.text_area(
    'Paragraph used for QA (you can also edit, or copy/paste new content)', 
    paragraph, 
    height=250
)
if written_passage:
    paragraph = written_passage

question = 'How many languages does bert understand?'
written_question = st.text_input(
    'Question used for QA (you can also edit, and experiment with the answers)', 
    question
)
if written_question:
    question = written_question

QA_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
QA_METHOD = 'POST'


if st.button('Run QA inference (get answer prediction)'):
    if paragraph and question:
        inputs = {'question': question, 'context': paragraph}
        payload = json.dumps(inputs)
        prediction = req.request(QA_METHOD, QA_URL, data=payload)
        answer = json.loads(prediction.content.decode("utf-8"))
        answer_span = answer["answer"]
        answer_score = answer["score"]
        st.write(f'Answer: **{answer_span}**')
        start_par = max(0, answer["start"]-86)
        stop_para = min(answer["end"]+90, len(paragraph))
        answer_context = paragraph[start_par:stop_para].replace(answer_span, f'**{answer_span}**')
        st.write(f'Answer context (and score): ... _{answer_context}_ ... (score: {format(answer_score, ".3f")})')
        st.write(f'Answer JSON: ')
        st.write(answer)
    else:
        st.write('Write some passage of text and a question')
        st.stop()


##########################################################
st.subheader('2. Information retrieval or sparse search')
##########################################################


client = Elasticsearch(BONSAI_URL)

question_4_similarity = 'How many languages did BERT understand in 2019?'
written_question_4_similarity = st.text_input(
    'Question used for similarity metric (you can also edit, and experiment)', 
    question_4_similarity
)
if written_question_4_similarity:
    question_4_similarity = written_question_4_similarity


st.write('Similarity is be computed relative to the documents in the bonsai mini-corpus with a `more like this` ES query.') 

similarity_list_size_bonsai = 1
written_similarity_list_size_bonsai = st.slider(
    'Pick how many similarity candidates to consider', 
    value=3, min_value=2, max_value=4
)
if written_similarity_list_size_bonsai:
    similarity_list_size_bonsai = written_similarity_list_size_bonsai

if st.button('Run sparse information retrieval (get relevance prediction in bonsai index)'):
    if question_4_similarity:
        query = {
            "more_like_this" : {
                "like" : question_4_similarity,
                "fields" : ["text"],
                "min_term_freq" : 1.9, 
                "min_doc_freq" : 4, 
                "max_query_terms" : 50,
            }
        }
        result = client.search(index="bert", body={"query": query})
        st.write(f'Most relevan passages (as candidates to contain the right answer) are: ')
        
        bonsai_hits = result['hits']['hits']
        docs_dict = [
            { 
                'score': hit['_score'], 
                **hit['_source'], 
                'question': question_4_similarity, # 'id': hit['_id'], 
            } 
            for hit in bonsai_hits
        ]

        df = pd.DataFrame.from_dict(docs_dict[0:similarity_list_size_bonsai])
        st.dataframe(df.style.highlight_max(axis=0), 1600, 300)
        st.write(bonsai_hits[0:similarity_list_size_bonsai]) 
        st.table(data=df)
    else:
        st.write('Write a question to compute similarity')
        st.stop()



with st.beta_expander("Formal definition of TF-IDF (click to see the details)"):
    st.write("""
        The TF-IDF (term frequency inverse document frequency) is a metric of 
        similarity defined by:
    """)
    st.latex(
        r'''\mathrm{tfidf} (t,d,D)={\frac {f_{t,d}}{\sum_{t'\in d}{f_{t',d}}}} \cdot '''
        r''' \log \frac{N}{1+|\{d \in D: t \in d\}|}'''
    )
    st.write("""where:  """)
    st.latex(
        r'''f_{t,d} \textrm{ is the the frequency of term $t$ in document $d$}\\'''
        r'''t' \textrm{ ranges over all terms $t'$ in $d$ different from term $t \neq t'$ }\\'''
        r'''N = |D| \textrm{ is the number $N$ of documents in the corpus $D$, and }\\'''
        r'''|\{d \in D: t \in d\}| \textrm{ is the number of documents in corpus $D$ where $t$ occurs}'''
    )



st.subheader('Customizing the topic for relevance metrics')
###########################################################

question_4_similarity_custom = 'How many attention heads does a transformer model have?'
written_question_4_similarity_custom = st.text_input(
    'Question used for similarity metric in custom topic (you can also edit, and experiment)', 
    question_4_similarity_custom
)
if written_question_4_similarity_custom:
    question_4_similarity_custom = written_question_4_similarity_custom

st.write('This passage will be added to the ones from mini-corpus and TFIDF will be run with `sklearn` vectorizer.') 

WIKI_TRANS = "&titles=Transformer_(machine_learning_model)"

response = req.request(WIKI_METHOD, f'{WIKI_URL}{WIKI_QUERY}{WIKI_TRANS}')
resp_json = json.loads(response.content.decode("utf-8"))
wiki_transf = resp_json['query']['pages']['61603971']['extract']
paragraph_4_tfidf = wiki_transf 

written_passage_4_tfidf = st.text_area(
    'Paragraph used for sparse information retrieval (you can also edit, or copy/paste new content)', 
    paragraph_4_tfidf, 
    height=250
)
if written_passage:
    paragraph_4_tfidf = written_passage_4_tfidf

similarity_list_size = 1
written_similarity_list_size = st.slider(
    'Pick how many similarity candidates to consider', 
    value=3, min_value=1, max_value=5
)
if written_similarity_list_size:
    similarity_list_size = written_similarity_list_size

def extract_score(entry): 
    return format(entry[1], ".3f")
def extract_text(entry, corpus, documents_tuples): 
    return corpus[documents_tuples[entry[0]][0]]
def extract_name(entry, documents_tuples): 
    return documents_tuples[entry[0]][0]
def question_relevance_dict(entry, corpus, documents_tuples):
    return {
        'score': extract_score(entry),
        'name': extract_name(entry, documents_tuples),
        'text': extract_text(entry, corpus, documents_tuples)
    }
def most_similar(corpus, question, n=1):
    documents_tuples = [document_tuple for document_tuple in corpus.items()]
    documents_texts = [document_text for document_text in corpus.values()]
    documents_texts.append(question)
    query_index = len(documents_texts) - 1
    tfidf_matrix = TfidfVectorizer().fit_transform(documents_texts)
    pairwise_similarity = tfidf_matrix * tfidf_matrix.T
    similarity_matrix = pairwise_similarity.toarray()
    np.fill_diagonal(similarity_matrix, 0)
    question_similarity_vector = list(enumerate(similarity_matrix[query_index]))
    question_most_similar = sorted(question_similarity_vector, key=lambda x: x[1], reverse=True)
    question_most_relevant = question_most_similar[0:n]
    question_relevance_list = [
        question_relevance_dict(entry, corpus, documents_tuples) 
        for entry in question_most_relevant
    ]
    return question_relevance_list

if st.button('Run similarity ranking (get relevance prediction within paragraph and bonsai index)'):
    if question_4_similarity_custom and paragraph_4_tfidf:
        response = client.search(index="bert", doc_type="_doc", body = {
            'size' : 10000,
            'query': { 'match_all' : {}}
        })
        corpus = dict([
            (hit['_source']['name'], hit['_source']['text']) 
            for hit in response['hits']['hits']
        ])
        corpus.update({'custom_document': paragraph_4_tfidf})
        relevance_list = most_similar(corpus, question_4_similarity_custom, similarity_list_size)
        relevance_list = [
            {**doc, 'question': question_4_similarity_custom} 
            for doc in relevance_list
        ]

        st.write(f'Most relevan passages (as candidates to contain the right answer) are: ')

        df_custom = pd.DataFrame.from_dict(relevance_list)
        st.dataframe(df_custom.style.highlight_max(axis=0), 1600, 300)
        st.write(relevance_list) 
        st.table(data=df_custom) 
    else:
        st.write('Write a paragraph and a question to compute similarity')
        st.stop()

###############################################################
st.subheader('3. Dense paragraph retrieval or semantic search')
###############################################################

# TODO: add dense search section
