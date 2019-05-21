#!/usr/bin/env python

'''
Generates an AST_09T-STF product
'''

from __future__ import print_function
import os
import json
import urllib3
import dateutil.parser
import requests
import numpy as np
from hysds.celery import app
import run_eval

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROD_SHORT_NAME = 'AST_09T-STF'
VERSION = "v1.0"
# determined globals
PROD = "{}-{}-{}" # eg: AST_L1T-20190514T341405_20190514T341435-v1.0
ALLOWED_INPUT_TYPES = ['AST_L1T']
INDEX = 'grq_{}_{}'.format(VERSION, PROD_SHORT_NAME)
PRODUCT_EXTENSION = 'HDF'
OUTPUT_FILENAME = '().tif'

def main():
    '''Generates the product if is not on GRQ'''
    # load parameters
    ctx = load_context()
    metadata = ctx.get("prod_metadata", False)
    prod_type = ctx.get("prod_type", False)
    input_prod_id = ctx.get("prod_id", False)
    if not prod_type not in ALLOWED_INPUT_TYPES:
        raise Exception("input needs to be of type: {}. Input is of type: {}".format(', '.join(ALLOWED_INPUT_TYPES), prod_type))
    starttime = ctx.get("starttime", False)
    endtime = ctx.get("endtime", False)
    location = ctx.get("location", False)
    #ingest the product
    generate_product(input_prod_id, starttime, endtime, location, metadata)

def generate_product(input_prod_id, starttime, endtime, location, metadata):
    '''determines if the product has been generated. if not, generates the product'''
    # generate product id
    prod_id = gen_prod_id(starttime, endtime)
    #get the input product path
    input_product_path = False
    for afile in os.listdir(input_prod_id):
        if afile.endswith(PRODUCT_EXTENSION.upper()) or afile.endswith(PRODUCT_EXTENSION.lower()):
            input_product_path = os.path.join(input_prod_id, afile)
    if not input_product_path:
        raise Exception('unable to find input hdf file in dir: {}'.format(input_prod_id))
    # determine if product exists on grq
    if exists(prod_id):
        print('product with id: {} already exists. Exiting.'.format(prod_id))
        return
    # make product dir
    if not os.path.exists(prod_id):
        os.mkdir(prod_id)
    output_fn = OUTPUT_FILENAME.format(prod_id)
    output_product_path = os.path.join(prod_id, output_fn)
    print('attempting to generate product: {}'.format(output_fn))
    # run product generation
    array = run_eval.run(input_product_path, output_product_path)
    if not os.path.exists(output_product_path):
        raise Exception('Failed generating product')
    dst, met = gen_jsons(prod_id, starttime, endtime, location, metadata)

    #fill additional values
    met['max_t'] = np.ma.max(array)
    met['99_percentile'] = np.percentile(array, 99)
    met['95_percentile'] = np.percentile(array, 95)
    met['99_delta'] = np.percentile(array, 99) - np.percentile(array, 50)
    met['95_delta'] = np.percentile(array, 99) - np.percentile(array, 50)
    met['median_t'] = np.ma.median(array)
    met['mean_t'] = np.ma.mean(array)
    met['std'] = np.ma.std(array)
    met['d_max'] = np.ma.max(array) - np.ma.median(array)

    # save the metadata fo;es
    save_product_met(prod_id, dst, met)
    # generate browse
    generate_browse(output_product_path, prod_id)

def gen_prod_id(starttime, endtime):
    '''generates the product id from the input metadata & params'''
    start = dateutil.parser.parse(starttime).strftime('%Y%m%dT%H%M%S')
    end = dateutil.parser.parse(endtime).strftime('%Y%m%dT%H%M%S')
    time_str = '{}_{}'.format(start, end)
    return PROD.format(PROD_SHORT_NAME, time_str, VERSION)

def exists(uid):
    '''queries grq to see if the input id exists. Returns True if it does, False if not'''
    grq_ip = app.conf['GRQ_ES_URL']#.replace(':9200', '').replace('http://', 'https://')
    grq_url = '{0}/{1}/_search'.format(grq_ip, INDEX)
    es_query = {"query": {"bool": {"must": [{"term": {"id.raw": uid}}]}}, "from": 0, "size": 1}
    return query_es(grq_url, es_query)

def query_es(grq_url, es_query):
    '''simple single elasticsearch query, used for existence. returns count of result.'''
    print('querying: {} with {}'.format(grq_url, es_query))
    response = requests.post(grq_url, data=json.dumps(es_query), verify=False)
    try:
        response.raise_for_status()
    except:
        # if there is an error (or 404,just publish
        return 0
    results = json.loads(response.text, encoding='ascii')
    #results_list = results.get('hits', {}).get('hits', [])
    total_count = results.get('hits', {}).get('total', 0)
    return int(total_count)

def generate_browse(product_path, prod_id):
    '''generates a browse from an input product path'''
    browse_path = os.path.join(prod_id, '{}.browse.png'.format(prod_id))
    browse_small_path = os.path.join(prod_id, '{}.browse_small.png'.format(prod_id))
    if os.path.exists(browse_path):
        return
    #conver to png
    os.system("convert {} -transparent black {}".format(product_path, browse_path))
    #convert to small png
    os.system("convert {} -transparent black -resize 300x300 {}".format(product_path, browse_small_path))

def gen_jsons(prod_id, starttime, endtime, location, metadata):
    '''generates ds and met json blobs'''
    ds = {"label": prod_id, "starttime": starttime, "endtime": endtime, "location": location, "version": VERSION}
    met = metadata
    return ds, met

def save_product_met(prod_id, ds_obj, met_obj):
    '''generates the appropriate product json files in the product directory'''
    if not os.path.exists(prod_id):
        os.mkdir(prod_id)
    outpath = os.path.join(prod_id, '{}.dataset.json'.format(prod_id))
    with open(outpath, 'w') as outf:
        json.dump(ds_obj, outf)
    outpath = os.path.join(prod_id, '{}.met.json'.format(prod_id))
    with open(outpath, 'w') as outf:
        json.dump(met_obj, outf)

def load_context():
    '''loads the context file into a dict'''
    try:
        context_file = '_context.json'
        with open(context_file, 'r') as fin:
            context = json.load(fin)
        return context
    except:
        raise Exception('unable to parse _context.json from work directory')

if __name__ == '__main__':
    main()

