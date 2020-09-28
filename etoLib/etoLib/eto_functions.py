#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import fsspec
import os
import rasterio as rio
from math import e
from osgeo import gdal, osr, gdal_array, gdalconst
import pandas as pd
import re
import earthpy.spatial as es
import xarray as xr
import numpy as np
import sys
import ogr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from contextlib import contextmanager 
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
import rioxarray
from shapely.geometry import Point, Polygon
import geopandas as gpd
from shapely.geometry import box
from fiona.crs import from_epsg
from matplotlib import pyplot as plt
from rasterio.plot import plotting_extent
import earthpy.plot as ep
import math

def lapply_brick(original_list,var_name,originalfolder,tiffolder,data_source,to_clip,bounds, pad_factor):
    
    # Returns list of open DatasetReader geoTIFFs from rasterio

    final_list=[]
    j=0
    for j in range(0, len(original_list)):
        
        if((data_source == 'METDATA') or (data_source == 'gridMET') or (data_source == 'Historical1915-LIVNEH')):
            
            fs = fsspec.filesystem('s3')
            fobj = fs.open(original_list[j])
            xds = xr.open_dataset(fobj)
            
            if((data_source == 'METDATA') or (data_source == 'gridMET')):
                orig_crs = xds.coordinate_system
            else:
                print('hello world')
            
            netcdf_var=xds[var_name]
            
#             if(data_source == 'LIVNEHhistorical1915'):
#                 netcdf_var.rio.set_crs('EPSG:4326', inplace=True)

            final = netcdf_var.assign_coords(lon=(((netcdf_var.lon + 180) % 360) - 180))
        
            final.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

            newfilename = original_list[j][:-3]+'.tif' # append .tif instead of .nc
            # os.chdir('/home/jupyter-rouze')
            os.chdir('.')


            if (to_clip == 'True'):
                
                lats = final.coords['lat'][:] 
                lons = final.coords['lon'][:]
                # lat_bnds, lon_bnds = [38.6, 42.54], [-76.3, -74.23] # originall hard-coded
                lat_bnds, lon_bnds = [bounds[0]-pad_factor, bounds[1]+pad_factor], [bounds[2]-pad_factor, bounds[3]+pad_factor]
                
                lat_inds = np.where((lats > lat_bnds[0]) & (lats < lat_bnds[1]))[0]
                lon_inds = np.where((lons > lon_bnds[0]) & (lons < lon_bnds[1]))[0]
                
                subset_netcdf=final[:,lat_inds,lon_inds]
                subset_netcdf.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

                local_file = 'in/temp/'+ original_list[0][-61:][:-3]+'.tif'

                bucket = original_list[0][0:11]

                bucket_filepath = original_list[0][12:]

#                 os.getcwd()
            
                subset_netcdf.rio.to_raster(local_file)
                
#                 s3_push_delete_local(local_file, bucket, bucket_filepath)
                
            else:

                final.rio.to_raster(newfilename, driver='GTiff')

        else:
            print('Hello world')
            
        
#         fs.close(fobj)
#     final_list.append(rio.open(fs.open(local_file)))
    final_list.append(rio.open(local_file))
        
    return final_list, local_file


# In[ ]:


def grepfxn(pattern,files):
    # Purpose: this function 
    precip_asc_df2=pd.DataFrame(files)
    idx_asc_in2 = []
    for i, precipfile in precip_asc_df2.iterrows():
        val = re.findall(pattern, precipfile.iloc[0])
        if len(val) > 0:
            idx_asc_in2.append(i)
            
    ascfileround2 = [files[i] for i in idx_asc_in2]
    return ascfileround2


# In[4]:


import boto3
def s3_push_delete_local(local_file, bucket, bucket_filepath):
        """
        This function will move the model outputs from a local folder to a cloud bucket.
        :param local_file: path the the local geo file
        :param outpath: path of a directory to be created in the cloud bucket
        :param bucket: name of the cloud bucket = 'dev-et-data'
        :param bucket_folder: "folder" in cloud bucket  = 'v1DRB_outputs'
        :return:
        """

        s3 = boto3.client('s3')
        with open(local_file, "rb") as f:
            print(bucket, bucket_filepath)
            s3.upload_fileobj(f, bucket, bucket_filepath)
        os.remove(local_file)


# In[ ]:


# function to get unique values 
def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 

    return(unique_list)    


# In[ ]:


def resample_raster_write(raster, name,scale=0.5):
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read(1)
    
    with rio.open(name, 'w', **profile) as dst:
        dst.write(data, 1)


# In[ ]:


def reproject_raster(src_filename, match_filename,dst_filename):
    
    from osgeo import gdal, gdalconst
    
    # Source
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    
    # We want a section of source that matches this:
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    
    # Output / destination
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    
    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    
    del dst # Flush


# In[ ]:


def rastermath(ensemble_list, iteration):
    
    # Purpose: To calculate the mean across X ensembles for each ith pixel
    
    # Returns a tuple with the first element the array, and the second the metadata of the the geoTiff file
    
    array_list=[]

    for j in range(0, len(ensemble_list)):
        dataset = ensemble_list[j]
        
        raster_array_read = dataset.read(iteration+1, masked=False) # fixed from masked=False to masked=True on 8/17/2020 - this applied to the gridMET data
        raster_meta = dataset.profile
        
#         raster_array_rot = np.rot90(raster_array_read,2)
#         raster_array_flip = np.flip(raster_array_rot,1)
        
#         array_list.append(raster_array_flip)
        array_list.append(raster_array_read)

    if(len(array_list) > 1):
        # Hard coded here by bands, might need to update this to adapt to any number of bands
        if(len(array_list) == 6):
            array_mean = np.mean( np.array([ array_list[0], array_list[1],array_list[2],array_list[3],array_list[4],array_list[5]]), axis=0 )
        elif len(array_list) == 5:
            array_mean = np.mean( np.array([ array_list[0], array_list[1],array_list[2],array_list[3],array_list[4]]), axis=0 )
        elif len(array_list) == 4:
            array_mean = np.mean( np.array([ array_list[0], array_list[1],array_list[2],array_list[3]]), axis=0 )
        elif len(array_list) == 3:
            array_mean = np.mean( np.array([ array_list[0], array_list[1],array_list[2] ]), axis=0 )
        else:
            array_mean = np.mean( np.array([ array_list[0], array_list[1]]), axis=0 )
    else:
        array_mean = array_list[0]
        
    return array_mean, raster_meta


# In[ ]:


def write_geotiff(data,meta,var_name,doy,year,folder):
    
    # Now we need to convert all of these arrays back to rasterio geoTIFFs
    
    os.chdir(folder+'/'+var_name)
    
    filename = var_name +'_' + str(year) +  str(doy).zfill(3)+'.tif'
    with rio.open(filename, 'w', **meta) as dst:
        dst.write(data, 1)
        
    return filename


# In[ ]:


class ET0_PM:
    
    def __init__(self, inputs, ET0_method,ET0_winddat,ET0_crop, constants):
        
        self.precipitation = inputs[0]
        self.downwelling_radiation = inputs[1]
        self.airtemperature_min = inputs[2]
        self.airtemperature_max = inputs[3]
        self.windspeed_2m = inputs[4]
        self.vapor_saturated = inputs[5]
        self.vapor_actual = inputs[6]
        self.elevation = inputs[7]
        self.latitude = inputs[8]
        self.dayofyear = inputs[9]
        
        self.ET0_method = ET0_method
        self.ET0_winddat = ET0_winddat
        self.ET0_crop = ET0_crop
        
        self.alpha=constants[0]
        self.z0 = constants[1]

    
    lambda_const = 2.45
    Gsc = 0.082
    G = 0
    sigma = 4.903*10**-9

    def incoming_shortwave(self):
        
#         self.R_s_in = self.downwelling_radiation*10**-6 # get into MJ m-2 day-1 (instead of J m-2 day-1)
        # self.Ta = (self.airtemperature_max+ self.airtemperature_min) / 2 - 273.15
        # self.delta = 4098 * (0.6108 * np.exp((17.27 * self.Ta)/(self.Ta+237.3))) / ((self.Ta + 237.3)**2) # slope of vapour pressure curve (S2.4), kPa C-1, http://www.fao.org/3/X0490E/x0490e07.htm
        self.R_s_in = self.downwelling_radiation*10**-6 * 3600 * 24 # From W m-2 to MJ m-2 day-1
        
    def outgoing_shortwave(self):
        
        self.R_s_out = self.alpha * self.R_s_in

    def outgoing_longwave(self):
        
        self.Ta = (self.airtemperature_max+ self.airtemperature_min) / 2 - 273.15  # Equation S2.1 in Tom McMahon's HESS 2013 paper, which in turn was based on Equation 9 in Allen et al, 1998.
        self.P = 101.3 * ((293 - 0.0065 * self.elevation) / 293)**5.26 # atmospheric pressure (S2.10), in kPa
        self.delta = 4098 * (0.6108 * np.exp((17.27 * self.Ta)/(self.Ta+237.3))) / ((self.Ta + 237.3)**2) # slope of vapour pressure curve (S2.4), kPa C-1, http://www.fao.org/3/X0490E/x0490e07.htm
        self.gamma = 0.00163 * self.P / self.lambda_const # psychrometric constant (S2.9) kPa C-1
        self.d_r2 = 1 + 0.033*np.cos(2*math.pi/365 * self.dayofyear) # dr is the inverse relative distance Earth-Sun (S3.6)
        self.delta2 = 0.409 * np.sin(2*math.pi/365 * self.dayofyear - 1.39) # solar dedication (S3.7)
        self.w_s = np.arccos(-np.tan(self.latitude) * np.tan(self.delta2))  # sunset hour angle (S3.8)
        self.N = 24/math.pi * self.w_s # calculating daily values
        self.R_a = (1440/math.pi) * self.d_r2 * self.Gsc * (self.w_s * np.sin(self.latitude) * np.sin(self.delta2) + np.cos(self.latitude) * np.cos(self.delta2) * np.sin(self.w_s)) # extraterristrial radiation (S3.5)
        self.R_so = (0.75 + (2*10**-5) * self.elevation) * self.R_a # clear sky radiation (S3.4)
        self.R_nl = self.sigma * ((self.airtemperature_max)**4 + (self.airtemperature_min)**4)/2 *(0.34 - 0.14 * np.sqrt(self.vapor_actual)) * (1.35 * self.R_s_in / self.R_so - 0.35) # estimated net outgoing longwave radiation (S3.3)

    def net_radiation(self):
        
        self.R_nsg =  self.R_s_in - self.R_s_out# net incoming shortwave radiation (S3.2)
        self.R_ng = self.R_nsg - self.R_nl # net radiation (S3.1)
        
    def ET0_calcs(self):

        if (self.ET0_crop == "short"):
            
            self.r_s = 70 # will not be used for calculation - just informative
            self.CH = 0.12 # will not be used for calculation - just informative
          
            # below in units of mm day-1, Final computation below
#             self.ET0_Daily_numerator = ( (0.408 * self.delta * (self.R_ng - self.G)) + 
#                                         (self.gamma * (900/(self.Ta + 273.2)) * self.windspeed_2m * 
#                                          (self.vapor_saturated - self.vapor_actual) ) ) 
             
#             self.ET0_Daily_denominator = (self.delta + self.gamma * (1 + 0.34*self.windspeed_2m )) # FAO-56 reference crop evapotranspiration from short grass (S5.18)
#             self.ET0_Daily = self.ET0_Daily_numerator/ self.ET0_Daily_denominator
            self.ET0_Daily_numerator1 = ( (0.408 * self.delta * (self.R_ng - self.G))) 
            
            self.ET0_Daily_numerator2 =  (self.gamma * (900/(self.Ta + 273.2)) *                                           self.windspeed_2m * (self.vapor_saturated -                                                               self.vapor_actual) )  
             
            self.ET0_Daily_denominator = (self.delta + self.gamma * (1 + 0.34*self.windspeed_2m )) # FAO-56 reference crop evapotranspiration from short grass (S5.18)
            self.ET0_Daily = (self.ET0_Daily_numerator1+self.ET0_Daily_numerator2)/ self.ET0_Daily_denominator
        
        else:
            
            self.r_s = 45 # will not be used for calculation - just informative
            self.CH = 0.50 # will not be used for calculation - just informative
            self.ET0_Daily_numerator = (0.408 * self.delta * (self.R_ng - self.G) + 
                                        (self.gamma * 1600 * self.u2 * (self.vs - self.va))/(self.Ta + 273))
            self.ET0_Daily_denominator = (self.delta + self.gamma * (1 + 0.38*self.u2))
            self.ET0_Daily = self.ET0_Daily_numerator / self.ET0_Daily_denominator # ASCE-EWRI standardised Penman-Monteith for long grass (S5.19)
        
        self.ET0_Daily = np.round(self.ET0_Daily, 2)
        self.ET0_Daily[self.ET0_Daily < 0] = 0
        
        return self.ET0_Daily


# In[ ]:


def aggregate_raster_inmem(raster, scale=0.5):
    t = raster.transform

    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = int(raster.height * scale)
    width = int(raster.width * scale)

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read( # Note changed order of indexes, arrays are band, row, col order not row, col, band
            out_shape=(raster.count, height, width),
            resampling=Resampling.bilinear)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            return dataset, profile  # Note yield not return     


# In[ ]:


def atmospheric_pressure(elevation):
    # https://en.wikipedia.org/wiki/Barometric_formula
    Pb = 101325 # Pa
    g0 = 9.80665
    M = 0.0289644
    hb = 0
    R = 8.3144598
    Tb = 288.15
    Lb = -0.0065 
    
    # https://www.mide.com/air-pressure-at-altitude-calculator
    # https://en.wikipedia.org/wiki/Atmospheric_pressure
    P = Pb * (1 + Lb/Tb*(elevation-hb))**(-(g0*M)/(R*Lb))
  
    return(P)


# In[ ]:


def relative_fromspecific(pressure, temperature, specifichumidity):
    # https://earthscience.stackexchange.com/questions/2360/how-do-i-convert-specific-humidity-to-relative-humidity
    test=(17.67*(temperature - 273.16))/(temperature - 29.65)
    rh = 0.263 * pressure * specifichumidity * np.exp(test)**-1
    
    return(rh)


# In[ ]:




