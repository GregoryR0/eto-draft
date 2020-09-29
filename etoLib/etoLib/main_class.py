import os
import fsspec
from itertools import chain
import pandas as pd

from .log_logger import log_make_logger

from etoLib.eto_functions import grepfxn, unique, lapply_brick, rastermath


def _make_sure_folder_exists(my_folder):

    if not os.path.exists(my_folder):
        os.makedirs(my_folder)


def _fake_write_geotiff(data,meta,var_name,doy,year,folder):
    
    # Now we need to convert all of these arrays back to rasterio geoTIFFs
    
    #os.chdir(folder+'/'+var_name)
    
    #filename = var_name +'_' + str(year) +  str(doy).zfill(3)+'.tif'
    _make_sure_folder_exists(folder)
    filename = folder+'/'+var_name +'_' + str(year) +  str(doy).zfill(3)+'.tif'
#     with rio.open(filename, 'w', **meta) as dst:
#         dst.write(data, 1)
        
    print("Fake Write:", filename)
    return filename


def _read_config(configurationfile):
    '''Read configuration file and parse out the inputs line by line'''

    # Note: if you want run rcp 8.5, then all you have to do is change the rcp_source parameter from within config file
    # It only affects grepfxn(rcp_source,all_files) below

    with open(configurationfile) as f:
        data = {}
        for line in f:
            key, value = line.strip().split(' = ')
            data[key] = value
    return(data)




# Note that the difference between historical and future outputs in cloud are based on these 2 configuration files.
# configurationfile = 'configurationfile_referenceET_test_future.ini'

class ETO:

    def __init__(self):

        my_folder='./log'
        _make_sure_folder_exists(my_folder)
        self.log=log_make_logger('ETO')

    def run_eto_ppt_temp(self):
        '''This is needed to retrieve the netCDF files from the dev-et-data AWS bucket'''

        configurationfile = 'configurationfile_referenceET_test_historical.ini'

        self.log.info(f'reading the config file {configurationfile}')

        data = _read_config(configurationfile)

        print(data)


        model_files = data['model_files']
        data_source = data['data_source']
        output_folder = data['output_folder']
        elevfile = data['elevfile']
        tiffolder = data['tiffolder']
        ET0_method = data['ET0_method']
        ET0_winddat = data['ET0_winddat']
        ET0_crop = data['ET0_crop']
        to_clip = data['to_clip']
        model = data['model']
        northmost = float(data['northmost'])
        southmost = float(data['southmost'])
        westmost = float(data['westmost'])
        eastmost = float(data['eastmost'])
        pad_factor = float(data['pad_factor'])
        rcp_source = data['rcp_source']
        MACA_start_bucket = data['MACA_start_bucket']

        # os.chdir(model_files)
        self.log.info(f'locating MACA file list bucket {MACA_start_bucket}')
        fs = fsspec.filesystem(model_files, anon=False, requester_pays=True)

        all_files = fs.find(MACA_start_bucket)

        # This prints all of the files in dev-et-data/in/DelawareRiverBasin/ or MACA_start_bucket...a big set of outputs, so skipped
        # print(all_files)


        # THE CODE BELOW IS PARSED FROM THE CONDIITION WHEN DEALING WITH METDATA


        # Split models apart that are to be used for ensemble averaging
        models_parsed = [x.strip() for x in model.split(',')]

        # Whittle down the number of files if the folder contains both rcp 4.5 and rcp 8.5 files
        # Right now, the code can only handle one model of METDATA output (8/21/2020)
        rcp_all_files = [grepfxn(rcp_source,all_files)][0]

        # Iterate the files by each each specified model
        self.log.info(f'Whittle by model MACA file list {models_parsed}')
        models_list=[]
        for i in range(len(models_parsed)):
            model_files_loop = [grepfxn(models_parsed[i],rcp_all_files)][0]
            models_list.append(model_files_loop)
                
        # Flatten series of lists into one list
        rcp_all_files = list(chain(*models_list))

        # prints all netCDF files from 1950-2100 from MACA (radiation, precipitation, wind etc.)
        print(rcp_all_files[0])


        # Find and compile the year blocks into a list
        dfis=[]
        for out in rcp_all_files:
            a=out.split('_')
            dfi = a[5]+'_'+a[6]
            dfis.append(dfi)

        # print(dfis)
            
        # Distill the above list into unique year blocks, as there will be duplicates from multiple climate inputs
        year_all=unique(dfis);print(year_all)

        # For prototyping only
        year_block=0
        # print(year_all)
        # Print the first entry in the year list
        print(year_all[year_block])


        # loop by each block associated with the MACA netCDF file naming structure
        for year_block in range(0,len(year_all)):

            year_block_files = grepfxn(year_all[year_block],rcp_all_files)
            
            print(year_block_files)
            self.log.info(f'year block = {year_block_files}')

            bounds=[southmost,northmost,westmost,eastmost]

            # precipitation
            self.log.info(f'lapply_brick preciptation {model_files}')
            rcp_pr = lapply_brick(grepfxn("pr",year_block_files), 'precipitation', model_files,tiffolder,data_source,to_clip=to_clip,bounds=bounds,pad_factor=pad_factor)
            # maximum air temperature
            self.log.info(f'lapply_brick tasmax {model_files}')
            rcp_tasmax = lapply_brick(grepfxn("tasmax",year_block_files), 'air_temperature', model_files,tiffolder,data_source,to_clip=to_clip,bounds=bounds,pad_factor=pad_factor)
            # minimum air temperature
            self.log.info(f'lapply_brick tasmin {model_files}')
            rcp_tasmin = lapply_brick(grepfxn("tasmin",year_block_files), 'air_temperature', model_files,tiffolder,data_source,to_clip=to_clip,bounds=bounds,pad_factor=pad_factor)
            
            start_year=year_all[year_block][0:4]
            end_year=year_all[year_block][5:9]

            start=start_year+'-01-01'
            end=end_year+'-12-31'
            datetimes = pd.date_range(start=start,end=end)
        #     i=10
            
            for i in range(0,rcp_pr[0][0].count):

                doy_loop = pd.Period(datetimes[i],freq='D').dayofyear
                year_loop = pd.Period(datetimes[i],freq='D').year

                # step 1: extract ith band from the raster stack
                # step 2: stack those ith bands together
                # step 3: do raster mean math from step 2
                pr_stack=[]

                # Purpose: create stacks of variables individually - this is like brick in R
                pr_ensemble = []
                tasmax_ensemble = []
                tasmin_ensemble = []


                # should be 1 array for each variable (mean of x ensembles for a given doy)
                # rcp_pr[0][0].read(1, masked=False).shape
                rcp_pr_doy = rastermath(rcp_pr[0], i)
                rcp_tasmax_doy = rastermath(rcp_tasmax[0], i)
                rcp_tasmin_doy = rastermath(rcp_tasmin[0], i)

                # Compute tasavg
                rcp_tasavg_doy = (rcp_tasmax_doy[0] + rcp_tasmin_doy[0])/2

                rcp_pr_doy[1]['count']=1
                rcp_tasmin_doy[1]['count']=1
                rcp_tasmax_doy[1]['count']=1

                rcp_pr_doy[1]['count']=1


                gTIFF_filename_ppt = _fake_write_geotiff(data=rcp_pr_doy[0],meta=rcp_pr_doy[1],var_name='PPT',doy=doy_loop,year=year_loop,folder=output_folder)

                gTIFF_filename_tasavg = _fake_write_geotiff(data=rcp_tasavg_doy,meta=rcp_tasmax_doy[1],var_name='Temp',doy=doy_loop,year=year_loop,folder=output_folder)
            
                print("Create tif Outputs", gTIFF_filename_ppt, gTIFF_filename_tasavg)

                '''Push newly created geoTIFF into specified bucket and its filepath'''

        #         os.chdir('.')
        #         s3_push_delete_local(output_folder+'/' + 'PPT' + '/' + gTIFF_filename_ppt , 'dev-et-data', 
        #                              'inTest/DelawareRiverBasin/PPT/'+ str(year_loop)  + '/' + gTIFF_filename_ppt )

        #         s3_push_delete_local(output_folder+'/' + 'Temp' + '/' + gTIFF_filename_tasavg , 'dev-et-data', 
        #                              'inTest/DelawareRiverBasin/Temp/'+ str(year_loop)  + '/' + gTIFF_filename_tasavg )

