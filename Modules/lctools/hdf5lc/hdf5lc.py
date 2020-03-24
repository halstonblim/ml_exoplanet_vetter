#!/usr/bin/env python
import numpy as np
import h5py
import scipy as sp

import os
import sys
import subprocess
import pyfits as pyf
import logging

import tsig
from tsig.lightcurve import FilebasedLightCurve 

class HDFLightCurve(FilebasedLightCurve):
    IDENTIFIER = "\211HDF\r\n\032\n"
    def __init__(self, name=''):
        super(HDFLightCurve, self).__init__(name)
        return 
    @staticmethod
    def istype(identifier):
        if identifier == self.IDENTIFIER:
            return True
        else:
            return False

    def load_from_file(self, label='all', ap=-1, rawmagkey='RawMagnitude', dmagkey='COSSEGMagnitude'):
        f = h5py.File(self.name, "r")
        appgroup =  f["LightCurve"]["AperturePhotometry"]
        self.data["jd"] = np.array(f["LightCurve"]["BJD"])
        self.data["cadence"] = np.array(f["LightCurve"]["Cadence"])
        self.data["bg"] = np.array(f["LightCurve"]["Background"]["Value"])
        self.data["bgerr"] = np.array(f["LightCurve"]["Background"]["Error"])
        try:
            self.data["flag"] = np.array(f["LightCurve"]["QFLAG"]) 
        except KeyError:
            pass
        # FIXME: for now, load_from_file by default load from bestaperture
        if ap == -1:
            bestap = appgroup.attrs["bestap"]
            api = appgroup["Aperture_%.3d" % bestap]
        else:
            api = appgroup["Aperture_%.3d" % ap]
        self.data["rlc"] = np.array(api[rawmagkey])
        self.data["rlcerr"] = np.array(api[rawmagkey+"Error"])
        self.data["x"] = np.array(api["X"])
        self.data["y"] = np.array(api["Y"])
        if dmagkey == 'COSSEGMagnitude':
            if "COSSEGMagnitude" in list(api.keys()):
                self.data["ltflc"] = np.array(api["COSSEGMagnitude"])
            elif "COSMagnitude" in list(api.keys()):
                self.data["ltflc"] = np.array(api["COSMagnitude"])
            else:
                self.data["ltflc"] = np.zeros(len(self.data["rlc"]))
        else:
            self.data["ltflc"] = np.array(api[dmagkey])
        return

    def read_app_lc(self, ap, kind='RawMagnitude'):
        f = h5py.File(self.name, "r")
        appgroup =  f["LightCurve"]["AperturePhotometry"]
        api = appgroup["Aperture_%.3d" % ap]
        try:
            mag= np.array(api[kind])
        except KeyError:
            f.close()
            raise 
        f.close()
        return mag

    def write_app_lc(self, ap, mag, kind='RawMagnitude'):
        f = h5py.File(self.name, "r+")
        appgroup =  f["LightCurve"]["AperturePhotometry"]
        api = appgroup["Aperture_%.3d" % ap]
        if kind in list(api.keys()):
            api[kind][:] = mag
        else:
            api.create_dataset(kind, data=mag)
        f.close()
        return

    def write_app_rms(self, ap, stats, kind="raw_rms"):
        f = h5py.File(self.name, "r+")
        appgroup =  f["LightCurve"]["AperturePhotometry"]
        api = appgroup["Aperture_%.3d" % ap]

        api.attrs[kind] = stats

        f.close()
        return 
    def write_to_file(self, outfile, replace=True, outheader="jd rlc"):
        # outheader indicates the sets to update in hdf5 files
        f = h5py.File(self.name, "r+")
        appgroup =  f["LightCurve"]["AperturePhotometry"]
        
        #FIXME: only update the best ap for now
        bestap = appgroup.bestap
        api = appgroup["Aperture_%.3d" % bestap]
        api["RawMagnitude"] = self.data["rlc"] 
        if 'ltflc' in outheader:
            api["COSMagnitude"] = self.data["ltflc"] 
        if 'gplc' in outheader:
            api["GPMagnitude"] = self.data["gplc"] 
        if 'epdlc' in outheader:
            api["EPDMagnitude"] = self.data["epdlc"] 

        return 

    def get_naps(self):
        f = h5py.File(self.name, "r")
        naps = f["LightCurve"]["AperturePhotometry"].attrs["naps"]
        return naps
    
    def write_example(self, outfile, baseline=100):
        f = h5py.File(outfile, "w")
        catgroup=f.create_group("CatalogueMagnitudes")
        lcgroup=f.create_group("LightCurve")
        f.attrs['TIC ID']= 123456
        f.attrs['RA'] = 120.0 
        f.attrs['Dec'] = -22.0
        f.attrs['CCD'] = 1
        f.attrs['Camera'] = 1
        f.attrs['Sector'] = 1
        f.attrs['Quality Flag'] = 1
        f.attrs['BJDoffset'] = 0 
        #catgroup.attrs['TessMag'] = starinfo['TessMag']
        catgroup.attrs['TessMag'] = 11. 
       

        lcgroup.create_dataset("BJD",data=np.zeros(baseline))
        lcgroup.create_dataset("Cadence",data=np.zeros(baseline))
        lcgroup.create_dataset("X",data=np.zeros(baseline))
        lcgroup.create_dataset("Y",data=np.zeros(baseline))
        
        bggroup=lcgroup.create_group("Background")
        bggroup.create_dataset("Value",data=np.zeros(baseline))
        bggroup.create_dataset("Error",data=np.zeros(baseline))
        
        apgroup = lcgroup.create_group("AperturePhotometry")
        apgroup.attrs["bestap"] = 1
        apgroup.attrs["naps"] = 2
        for i in range(apgroup.attrs["naps"]):
            api = apgroup.create_group("Aperture_%.3d" % i)
            api.attrs["annulus"] = "12:10"
            api.attrs["radius"] = "2.5"
            # rms, mmd, cdpp_hr
            api.attrs["raw_rms"] = [0.001, 0.001, 0.001]
            api.attrs["epd_rms"] = [0.001, 0.001, 0.001]
            api.attrs["cos_rms"] = [0.001, 0.001, 0.001]
            api.attrs["gp_rms"] = [0.001, 0.001, 0.001]
            api.create_dataset("RawMagnitude",(baseline,),dtype='f')
            api.create_dataset("RawMagnitudeError",(baseline,),dtype='f')
            api.create_dataset("X",(baseline,),dtype='f')
            api.create_dataset("Y",(baseline,),dtype='f')
            api.create_dataset("QualityFlag",(baseline,),dtype='|S1')
            api.create_dataset("EPDMagnitude",(baseline,),dtype='f')
            api.create_dataset("COSMagnitude",(baseline,),dtype='f')
            api.create_dataset("GPMagnitude",(baseline,),dtype='f')
        f.close() 

    def read_attr(self, key):
        f = h5py.File(self.name, "r")
        if key in list(f.attrs.keys()):
            var = f.attrs[key]
            f.close()
            return var
        elif key in list(f["CatalogueMagnitudes"].attrs.keys()):
            var = f["CatalogueMagnitudes"].attrs[key]
            f.close()
            return var
        else:
            appgroup =  f["LightCurve"]["AperturePhotometry"]
            if key in list(appgroup.attrs.keys()):
                var = appgroup.attrs[key] 
                f.close()
                return var
            else:
                f.close()
                raise ValueError("hdf5 file %s does not contain attribute %s" % (self.name, key))
    def read_starinfo(self):
        starinfo={}
        starinfo["ticid"] = self.read_attr("TIC ID") 
        starinfo["ra"] = self.read_attr("RA") 
        starinfo["dec"] = self.read_attr("Dec") 
        starinfo["Tmag"] = self.read_attr("TessMag") 
        starinfo["Camera"] = self.read_attr("Camera")
        starinfo["CCD"] = self.read_attr("CCD")
        starinfo["Sector"] = self.read_attr("Sector") 
        starinfo["Flag"] = self.read_attr("Quality Flag")
        starinfo["BJDoffset"] = self.read_attr("BJDoffset") 
        return starinfo

    def load_basic_info(self):
        f = h5py.File(self.name, "r")
        appgroup =  f["LightCurve"]["AperturePhotometry"]
        self.data["jd"] = np.array(f["LightCurve"]["BJD"])
        self.data["cadence"] = np.array(f["LightCurve"]["Cadence"])
        self.data["bg"] = np.array(f["LightCurve"]["Background"]["Value"])
        self.data["bgerr"] = np.array(f["LightCurve"]["Background"]["Error"])
        self.data["x"] = np.array(f["LightCurve"]["X"])
        self.data["y"] = np.array(f["LightCurve"]["Y"])
        
        return 

    def write_attr(self, key, var):
        f = h5py.File(self.name, "r+")
        if key in list(f.attrs.keys()):
            f.attrs[key]=var
        elif key in list(f["CatalogueMagnitudes"].attrs.keys()):
            f["CatalogueMagnitudes"].attrs[key]=var
        else:
            appgroup =  f["LightCurve"]["AperturePhotometry"]
            if key in list(appgroup.attrs.keys()):
                appgroup.attrs[key] = var
            else:
                f.close()
                raise ValueError("hdf5 file %s does not contain attribute %s" % (self.name, var))
        f.close()



    def write_new(self, outfile, ticinfo):
        f = h5py.File(outfile, "x")
        catgroup=f.create_group("CatalogueMagnitudes")
        lcgroup=f.create_group("LightCurve")
        f.attrs['TIC ID']= ticinfo["ticid"] 
        f.attrs['RA'] = ticinfo["ra"] 
        f.attrs['Dec'] =  ticinfo["dec"]
        f.attrs['CCD'] = ticinfo["CCD"]
        f.attrs['Camera'] = ticinfo["Camera"]
        # FIXME: sector should be a list indicating the sectors with data.
        f.attrs['Sector'] = ticinfo["Sector"]
        f.attrs['Quality Flag'] = ticinfo["Flag"]
        f.attrs['BJDoffset'] = ticinfo["BJDoffset"] 
        #catgroup.attrs['TessMag'] = starinfo['TessMag']
        catgroup.attrs['TessMag'] = ticinfo["Tmag"] 
        f.close() 
    
    def write_basic_info(self, outfile, jdarr, naps, jdseq={'jd':0, 'cadence':1, 'x':2, 'y':3, 'bg':4, 'bgerr':5}):
        f = h5py.File(outfile, "r+")
        lcgroup = f["LightCurve"] 
        #print jdarr[jdseq['jd']]
        lcgroup.create_dataset("BJD",data=jdarr[jdseq['jd']])
        lcgroup.create_dataset("Cadence",data=jdarr[jdseq['cadence']])
        lcgroup.create_dataset("X",data=jdarr[jdseq['x']])
        lcgroup.create_dataset("Y",data=jdarr[jdseq['y']])
        
        # FIXME: background is the same for different apertures. 
        bggroup=lcgroup.create_group("Background")
        bggroup.create_dataset("Value",data=jdarr[jdseq['bg']])
        bggroup.create_dataset("Error",data=jdarr[jdseq['bgerr']])


        apgroup = lcgroup.create_group("AperturePhotometry")
        apgroup.attrs["naps"] = naps 
        apgroup.attrs["bestap"] = 1
        f.close() 

    def write_qflag(self, outfile, qflag_cadence, qflag_flag):
        f = h5py.File(outfile, "r+")
        lcgroup = f["LightCurve"] 
        cadence = lcgroup["Cadence"]
        index = np.in1d(qflag_cadence, cadence)
        lcgroup.create_dataset("QFLAG",data=qflag_flag[index])
        f.close()
    def write_aps_from_fiphot(self, outfile, aparray, lcarr, lcarrseq=None, dmagkey="COSMagnitude"):
        # aparray is the string containing aperture characteristics
        # lcarr is the data structure containing lc files, first index is nap
        #second index is raw mag, rawmagerr, x, y, flag 
        f = h5py.File(outfile, "r+")
        
        if lcarrseq is None:
            lcarrseq = np.arange(5)
        apgroup = f["LightCurve"]["AperturePhotometry"] 
        #baseline = lcarr[0].shape[0]
        baseline = lcarr.shape[0]
        for i in range(len(aparray[0])):
            # print "create_group Aperture_%.3d" % aparray[0][i]
            api = apgroup.create_group("Aperture_%.3d" % aparray[0][i])
            
            api.attrs["annulus"] = ":".join(aparray[1][i].split(":")[1:])  
            api.attrs["radius"] = aparray[1][i].split(":")[0]
            # place holder, need to be filled: rms, mmd, cdpp_hr
            api.attrs["raw_rms"] = [0.001, 0.001, 0.001]
            api.attrs["epd_rms"] = [0.001, 0.001, 0.001]
            api.attrs["cos_rms"] = [0.001, 0.001, 0.001]
            api.attrs["gp_rms"] = [0.001, 0.001, 0.001]

            #
            #api.create_dataset("RawMagnitude",data=lcarr[0][i, 0, :], chunks=True)
            #api.create_dataset("RawMagnitudeError",data=lcarr[0][i,1,:], chunks=True)
            #api.create_dataset("X",data=lcarr[0][i, 2, :], chunks=True)
            #api.create_dataset("Y",data=lcarr[0][i,3,:], chunks=True)
            
            #api.create_dataset("RawMagnitude",data=lcarr[0][i, :, 0], chunks=True)
            #api.create_dataset("RawMagnitudeError",data=lcarr[0][i,:, 1], chunks=True)
            #api.create_dataset("X",data=lcarr[0][i, :, 2 ], chunks=True)
            #api.create_dataset("Y",data=lcarr[0][i,:, 3], chunks=True)
            #api.create_dataset("QualityFlag",data=lcarr[1][i, :], chunks=True)
            #print type(np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[0])].values))
            #print np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[0])].values).shape
            api.create_dataset("RawMagnitude",data=list(np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[0])].values)), dtype=np.float64, chunks=True)
            api.create_dataset("RawMagnitudeError",data=list(np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[1])].values)), dtype='f', chunks=True)
            api.create_dataset("X",data=list(np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[2])].values)), dtype='f', chunks=True)
            api.create_dataset("Y",data=list(np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[3])].values)), dtype='f', chunks=True)
            qflag = np.array(lcarr.loc[:, "ap%d_%d" % (i,lcarrseq[4])].values, dtype='|S1')
            #print len(qflag), type(qflag)
            api.create_dataset("QualityFlag",data=qflag, chunks=True)
            
            #api.create_dataset("EPDMagnitude",(baseline,),dtype='f', chunks=True)
            api.create_dataset(dmagkey,(baseline,),dtype='f', chunks=True)
            #api.create_dataset("GPMagnitude",(baseline,),dtype='f', chunks=True)
        f.close() 




if __name__=='__main__':
    lc = HDFLightCurve()
    lc.write_example("temp.h5")