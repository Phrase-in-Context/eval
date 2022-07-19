""" A flask-based web server

Usage:
    python server.py --config ../config/dev.json


"""
import sys
sys.path.append("../")

import os
import math
import time
import json

from datasets import load_dataset
from retrieval_ranking import CreateLogger
from retrieval_ranking import ROOT_DIR
from system import System

from flask import Flask
from flask import request, render_template, jsonify
app = Flask(__name__, static_folder="static")


system = None


@app.route('/')
@app.route('/index')
def index():
    """ """

    # example_fpath = os.path.join(os.path.dirname(__file__), "./examples.json")
    # with open(example_fpath) as fp:
    #     examples = json.load(fp)

    # Prepare examples loaded from Huggingface
    pr_pass = load_dataset("PiC/phrase_retrieval", "PR-pass")
    examples = [example for example in pr_pass["test"]][:100]

    return render_template('index.html', examples=examples)


def _pick_top_n(results, top_n):
    """ """
    new_results = []
    closed = set()
    for result in results:
        if all([x not in result['phrase'] for x in closed]):
            # if the higher-scored phrases are not a substring of result['phrase']
            new_results.append(result)
            closed.add(result['phrase'])

        if len(new_results) >= top_n:
            break

    return new_results


@app.route('/answer', methods=['GET', 'POST'])
def answer():
    """ """
    global system
    global logger
    global model_fasttext
    global model_bert
    global model_sentbert
    global model_use

    start_time = time.time()

    text = request.form.get('paragraph_text', '').strip()
    query = request.form.get('question_text', '').strip()
    extractor_name = request.form.get('extractor', '').strip()
    scorer = request.form.get('scorer', '').strip()

    if scorer == 'fasttext':
        model = model_fasttext
    elif scorer == 'bert':
        model = model_bert
    elif scorer == 'sentbert':
        model = model_sentbert
    elif scorer == 'use':
        model = model_use
    else:
        logger.error("ERROR unsupported model: $s", scorer)

    filters = {
        'semantic_search': True,
        'acronym': request.form.get('acronym', '').strip() == "true",
        'date': request.form.get('date', '').strip() == "true",
        'name': request.form.get('name', '').strip() == "true",
        'number': request.form.get('number', '').strip() == "true",
    }
    
    logger.debug("extractor_name: %s", extractor_name)
    logger.debug("scorer: %s", scorer)
    logger.debug("filters: %s", filters)

    # system.set_ss_extractor(extractor_name)
    # system.set_ss_scorer(scorer_name, model_fpath, scorer_type)
    # system.set_text([text])
    # results = system.search(query, top_n=10, filters=filters)

    model.set_ss_extractor(extractor_name)
    model.set_text([text])
    results = model.search(query, top_n=10, filters=filters)

    results = _pick_top_n(results, 3)    


    end_time = time.time()
    diff_time = round((end_time - start_time)*1000)

    info = "[info] number of candidates: " + str(len(model.semantic_search.phrases)) +  '<br>'
    info = info + "[info] processing time (ms): " + str(diff_time) + '<br><br>'

    html = info + str(text)
        
    for i, result in enumerate(results, 1):
        score_str = "{:.4}".format(result['score'])
        html_snippet = '<span class="paragraph_sentence" data-toggle="tooltip" title="{}"><mark class="top-phrase" id="top-{}-phrase">{}</mark></span>'.format("score: " + score_str, i, result['phrase'])
        # print(result)
        # print(html_snippet)
        html = html.replace(result['phrase'], html_snippet)

    return jsonify({'html': html})


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """ """
    global logger
    global model_fasttext
    global model_bert
    global model_sentbert
    global model_use

    start_time = time.time()
    end_time = []

    #
    # config_fpath = os.path.join(ROOT_DIR, "./model_config.json")
    # with open(config_fpath) as f:
    #     config = json.load(f)
    
    text = request.form.get('paragraph_text', '').strip()
    query = request.form.get('question_text', '').strip()
    extractor_name = request.form.get('extractor', '').strip()
    
    # ToDO : receive from the interface
    list_scorer = ['fasttext', 'bert', 'sentbert', 'use']

    dict_rst = {}
    
    for scorer in list_scorer:

        if scorer == 'fasttext':
            model = model_fasttext
        elif scorer == 'bert':
            model = model_bert
        elif scorer == 'sentbert':
            model = model_sentbert
        elif scorer == 'use':
            model = model_use
        else:
            logger.error("ERROR unsupported model: $s", scorer)

        filters = {
            'semantic_search': True,
            'acronym': request.form.get('acronym', '').strip() == "true",
            'date': request.form.get('date', '').strip() == "true",
            'name': request.form.get('name', '').strip() == "true",
            'number': request.form.get('number', '').strip() == "true",
        }

        logger.debug("extractor_name: %s", extractor_name)
        logger.debug("scorer: %s", scorer)
        logger.debug("filters: %s", filters)

        model.set_ss_extractor(extractor_name)
        model.set_text([text])
        results = model.search(query, top_n=10, filters=filters)

        dict_rst[scorer] = _pick_top_n(results, 4)
        
        end_time.append( round((time.time() - start_time)*1000) )
        start_time = time.time()

    diff_time = sum(end_time)

    info = "[info] number of candidates: " + str(len(model.semantic_search.phrases)) +  '<br>'
    info = info + "[info] processing time (ms): " + str(diff_time) + " ("
    for time_model in end_time:
        info = info + str(time_model) + ", "
        
    info = info[:-2]
    info = info + ') <br><br>'

    html = info
    html += '<style> \
            tr{line-height:20px; border:1px solid blue;} \
            td{line-height: 20px; border:1px solid black; padding: 5px 10px 5px 15px;} \
            </style>'
    html += '<table width="80%"><tbody>'
    
    html = html + '<tr bgcolor=lightgrey><td>Model</td><td>' + query + '</td><td></td></tr>'
    for key in dict_rst.keys():
        html = html + '<tr><td>' + key + '</td>'

        html += '<td>'
        for i, predict in enumerate(dict_rst[key]):
            html = html + str(i+1) + '. ' + predict['phrase'] + '<br>'
        html += '</td>'
        
        html += '<td>'
        for i, predict in enumerate(dict_rst[key]):
            html = html + str(math.floor( predict['score'] * 10000 ) / 10000.0) + '<br>'
        html += '</td>'
                
        html += '</tr>' 

    return jsonify({'html': html})


def _load_model(scorer):

    config_fpath = os.path.join(ROOT_DIR, "./model_config.json")
    with open(config_fpath) as f:
        config = json.load(f)

    scorer_name = scorer.split(':')[0].strip()
    scorer_type = scorer.split(':')[1].strip()

    try:
        model_fpath = [x for x in config if x['scorer'] == scorer_name][-1]['model_fpath']
        # model_fpath = os.path.join(ROOT_DIR, model_fpath)
    except:
        logger.error("Unsupported scorer: %s", scorer_name)

    model = System()
    model.set_ss_scorer(scorer_name, model_fpath, scorer_type)

    return model


if __name__ == '__main__':

    logger = CreateLogger()

    # model_fasttext = _load_model('fasttext:')
    # model_bert = _load_model('transformers:bert')
    # model_sentbert = _load_model('sentbert:Sdistilroberta-base-msmarco-v2')
    # model_use = _load_model('use:v4')

    model_bert = _load_model('BERT:bert-base-uncased')
    app.run(debug=True, host='0.0.0.0', port=5007)
