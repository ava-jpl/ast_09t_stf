#!/usr/bin/env python

from __future__ import print_function
import os
import re
import math
import time
import argparse
import rasterio
import numpy as np
from osgeo import gdal
import tir_fit
import multiprocessing

GET_BANDS = ['10', '11', '12', '13', '14']
BAND_REGEX = 'HDF4_EOS:EOS_SWATH:.*(?:ImageData|SurfaceRadianceTIR:Band|TIR_Swath:ImageData)([0-9]{1,2})$'

def run(hdf_path, outpath):
    '''
    takes the input hdf path, generates a SO2 band map, and then saves the result to outpath
    '''
    array = gen(hdf_path)
    write_as_tif(array, outpath)
    return array

def gen(hdf_path):
    '''
    loads the hdf path, generates the campion array, and returns the result
    '''
    conversion = {}
    #the band scaling factors
    conversion = {'10': 0.006882, '11': 0.006780, '12': 0.006590, '13': 0.005693, '14': 0.005225}
    radiance = {}
    band_layers = []
    print('loading input file{}'.format(hdf_path))
    src_ds = gdal.Open(hdf_path)
    sub = src_ds.GetSubDatasets()
    for data in sub:
        subdataset = str(data[0])
        match = re.search(BAND_REGEX, subdataset)
        if not match:
            #print('{} not matching regex'.format(subdataset))
            continue
        band = match.group(1)
        field = match.group(0)
        if band not in GET_BANDS:
            print('passing band analysis...')
            continue
        print('loading band: {} from: {}'.format(match.group(1), subdataset))
        #print('loading band %s' % band)
        #https://lpdaac.usgs.gov/products/ast_09tv003/
        raster = gdal.Open(data[0]).ReadAsArray().astype('uint16')
        
        #mast out zero and max (saturated) values
        raster = np.ma.masked_where(raster <= 0, raster)
        raster = np.ma.masked_where(raster >= 3000, raster)
        raster = np.ma.multiply(raster.astype(np.float32), conversion[str(band)])
        band_layers.append(raster)


    band_stack = np.ma.dstack(band_layers)
    band_values = np.ma.multiply(band_stack, 8100 * math.pi) # convert to W/micron from W/m^2/str/um

    #init arrays for output
    w, h, z = band_values.shape
    tir_emittance = np.ma.zeros((w,h), np.float32)

    #run temp fitting asynchronously
    mask = np.ma.getmask(np.ma.sum(band_values, axis=2))
    indices = np.ndindex(w,h)
    start = time.time()
    print('starting temp fitting...')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = [pool.apply_async(get_temp, (band_values[i],i) ) for i in indices if not mask[i]]
    for result in results:
        t, index = result.get()
        tir_emittance[index] = t 
    pool.close()
    pool.join()
    end = time.time()
    tir_emittance = np.ma.masked_where(tir_emittance <=240, tir_emittance)
    print('total runtime: {0} minutes'.format((end - start)/60))
    #print_stats(tir_emittance)

    return tir_emittance

def get_temp(radiance_values, index):
    t = tir_fit.temp_fit(radiance_values)
    return t.run_one_temp_fitting(), index

def write_array(input_array, out_path):
    '''alternative way of saving array'''
    h,w = input_array.shape
    #print('width: %s, height: %s' % (w,h))
    with rasterio.open(out_path, 'w', driver='GTiff', width=w, height=h, count=1,
        dtype=rasterio.float64) as s:
        s.write(input_array, indexes=1)

def write_as_tif(input_array, outpath):
    '''
    write the input_array to the outpath
    '''
    if os.path.exists(outpath):
        os.remove(outpath)
    driver = gdal.GetDriverByName("GTiff")
    [cols, rows] = input_array.shape
    outdata = driver.Create(outpath, rows, cols, 1, gdal.GDT_Float64)
    outdata.GetRasterBand(1).WriteArray(input_array)
    #outdata.GetRasterBand(1).SetNoDataValue(0)
    outdata.FlushCache()

def parser():
    '''
    Construct a parser to parse arguments
    @return argparse parser
    '''
    parse = argparse.ArgumentParser(description="Generates product from input file")
    parse.add_argument("-f", "--hdf", required=True, help="path of input hdf file", dest="hdf")
    parse.add_argument("-o", "--out", required=True, help="path to output file", dest="out")
    return parse

if __name__ == '__main__':
    args = parser().parse_args()
    run(args.hdf, args.out)
