# This file contains code that is useful for analyzing raw acquisition data from
# CHIME and CHIME/outrigger telescopes. Examples for how to use the code to
# create plots are provided at the bottom.

import numpy as np
import dateutil
import datetime
import pytz
import time
import glob
import h5py
import re
from functools import reduce
import itertools  
import os
from matplotlib.backends.backend_pdf import PdfPages
import ephem
import requests
from bs4 import BeautifulSoup
from scipy.stats import kurtosis as scipy_kurtosis
import chime_frb_constants as constants
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

import logging
# Logging Config
LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s "
logging_format += "%(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
log = logging.getLogger()


# Create a class that reads the data in, and contains the functions for plotting

# bad_inputs : list of int
#    An Nx3 list specifying bad inputs to avoid. Each N entry is a coordinate: [crate, slot, input]
BAD_INPUTS = [
    [0, 0, 1],
    [0, 0, 3],
    [0, 0, 6],
    [0, 1, 6],
    [0, 1, 14],
    [0, 2, 6],
    [0, 2, 9],
    [0, 3, 0],
    [0, 3, 2],
    [0, 3, 6],
    [0, 3, 7],
    [0, 3, 15],
    [0, 4, 4],
    [0, 4, 5],
    [0, 4, 10],
    [0, 5, 3],
    [0, 5, 4],
    [0, 5, 7],
    [0, 5, 9],
    [0, 6, 3],
    [0, 6, 4],
    [0, 7, 6],
    [0, 7, 8],
    [0, 8, 2],
    [0, 8, 6],
    [0, 8, 15],
    [0, 9, 5],
    [0, 9, 11],
    [0, 10, 8],
    [0, 10, 10],
    [0, 12, 0],
    [0, 12, 3],
    [0, 12, 4],
    [0, 13, 4],
    [0, 14, 3],
    [0, 14, 4],
    [0, 14, 7],
    [0, 14, 10],
    [0, 15, 0],
    [0, 15, 4],
    [0, 15, 5],
    [0, 15, 9],
    [0, 15, 12],
    [0, 15, 15],
] # as of Oct 18, 2023 for GBO

class RawAcqException(Exception):                                                                                                                
    """                                                                                                                                                  
    Base Exception used in RawAcq                                                                                                               
    """                                                                                                                                                  
    pass

class RawAcq(object):
    """ A class for reading CHIME and CHIME/Outrigger raw ADC data and producing
    various useful diagnostic plots (timeseries, rms, FFT, dynamic spectrum, histogram, etc).
    
    In the default config for the raw acquisition server, 2 kB of raw acquisition data is saved
    every 30 seconds. Each .h5 file contains ~4 hours of data.
    
    Attributes
    ----------
    filenames : np.ndarray of str
        An array of filenames that will be loaded into the RawAcq object for plotting purposes
    dates : np.ndarray of datetimes
        An array of len=2 with UTC dates [start_date, end_date] indicating which filenames should be
        loaded into the RawAcq object for plotting purposes
    plot_dir : str
        The file path where diagnostic plots will be saved
    raw_acq_dir : str
        The file path where the raw acquisition data is being saved
    
     ===NOTE====
     Some explanation, so the following parameters are understandable.
     
     For every input, the ADC samples the voltage every 1.25 ns (1/800 MHz), and it quantizes 
     the value into bits with 8-bit resolution (a number between -128 and 127). The F-engine 
     packs those samples into a "frame" of 2048 samples (1.25*2048 = 2.56 us), then sends the 
     frame to the next stage. It does this in realtime.
     
     So, we get 1 frame every 2.56 mus. Then every 30 seconds, we take just one of those frames 
     and save it to disk. The same frame (corresponding to the same timestamp) for every input.
     Each h5 file is saved with metadata telling which FPGA crate the data is from (crate), 
     which ADC board within the crate (slot), and which input within each board (input), 
     along with the timestamps for each frame that it saves. The data from each crate, slot, input
     is loaded into the h5 file in a staggered way, so here we have to use the metadata to
     recover which packets belong to which crate, slot, input, and frame timestamp.
     
     "Least Significant Bits" : An ADC has a maximum range it can tolerate in volts (+- 250 mV for CHIME), 
     that it digitizes into 8 bits. The ADC splits that range into 256 levels (8 bits) that cover from 
     -250 to +250 mV (corresponding to -128 to 127). The LSB is the unit of the smallest bin, 
     corresponding to ~250/128 = 1.95 ~ 2 mV.
    ===========
    
    timestream : np.ndarray of int
        An array of the actual timestream of voltage values in LSB = Least Significant Bit units.
        Will have shape: [npackets, 2048], where npackets is all of the frames saved in crate, slot, 
        input, and time space, unraveled.
    crates : np.ndarray of int
        An array of metadata indicating what crate each frame in timestream corresponds to.
        Will have len=npackets.
    slots : np.ndarray of int
        An array of metadata indicating what fpga slot each frame in timestream corresponds to.
        Will have len=npackets.
    inputs : np.ndarray of int
        An array of metadata indicating what SMA input on the FPGA each frame in timestream corresponds to.
        Will have len=npackets.
    ctimes : np.ndarray of float
        An array of metadata indicating the ctime of each frame in timestream.
        Will have len=npackets.    
    fpga_counts : np.ndarray of int
        An array of metadata indicating the fpga timestamp of frame in timestream.
        Will have len=npackets.  
    start_time : datetime
        A datetime object indicating the UTC ctime of first frame
    end_time : datetime
        A datetime object indicating the UTC ctime of last frame
    ctime_frames : datetimes
        An array of metadata indicating the ctimes for each unique frame
        Will have len=number of frames that were saved
    num_crates : int
        Number of crates in this dataset
    num_slots : int
        Number of FPGA slots in this dataset
    num_inputs : int
        Number of SMA inputs in this dataset
    num_frames : int
        The number of unique frames in the dataset
        
    
    Methods
    -------
    read_data(filenames or datetimes)
    get_timestream_for_input(slot, input, crate)
    calc_rms(slot, input, crate)
    calc_kurtosis(slot, input, crate)
    calc_fft(slot, input, crate)
    calc_dyn_spec(slot, input, crate)
    get_timeseries(slot, input, crate)
    plot_all_inputs_diagnostic(params)
    plot_input_summary_diagnostic(params)
    plot_total_dynamic_spectrum(params)
    plot_slot_dynamic_spectrum(params)
    TODO: plot_readout_diagnostic(plot showing the readout of samples as a function of input, slot, ctime, etc)
    """
    def __init__(
        self,
        filenames: np.ndarray = None,
        dates: np.ndarray = None,
        plot_dir: str = '/home//bandersen/raw_acq_diagnostics/',
        raw_acq_dir: str = '/home/masuilab/data/raw_acq/',
    ):
        """ Instantiates a RawAcq object for a given set of files or date range. You must define
            one or the other!
        """
        self.raw_acq_dir = raw_acq_dir
        self.plot_dir = plot_dir
        
        # Check to make sure parameters are in the right form
        if not os.path.exists(plot_dir):
            raise RawAcqException(
                "Output directory does not exist: {}".format(plot_dir)
            )
        
        if not os.path.exists(raw_acq_dir):
            raise RawAcqException(
                "Raw acquisition directory does not exist: {}".format(raw_acq_dir)
            )
        
        if ((filenames is None) and (dates is None)) or \
           ((filenames is not None) and (dates is not None)):
            raise RawAcqException(
                "Define either filenames or dates to be read in."
            )
            
        if dates is not None:
            if len(dates) != 2 or \
               type(dates[0]) != datetime.datetime or \
               type(dates[1]) != datetime.datetime:
                raise RawAcqException(
                    "Define dates=[start_date, end_date] where start_date and end_date are datetimes."
                )
            
        if (filenames is not None):
            for fn in filenames:
                if not os.path.isfile(fn):
                    raise RawAcqException(
                        "File does not exist: {}".format(fn)
                    )
        
        load_fns = filenames is not None
        load_dates = dates is not None
        
        if load_fns:
            self.read_data(filenames = filenames)
            
        elif load_dates:
            self.read_data(dates = dates)
       
    
    def read_data(self, filenames: np.ndarray = None, dates: np.ndarray = None):
        """ This function reads in raw acquisition data based on either an array 
            of directly given filenames or a time range.
        
        Parameters
        ----------
        filenames : np.ndarray of str
            An array of filenames that will be loaded into the RawAcq object for plotting purposes.
        dates : np.ndarray of datetimes
            An array of len=2 with UTC dates [start_date, end_date] indicating which filenames should be
            loaded into the RawAcq object for plotting purposes
        """
        utc = pytz.timezone('UTC')
        # If dates provided, find filenames within those dates
        if dates is not None:
            start_date = dates[0]
            end_date = dates[1]
            
            log.info("Reading in data for dates between start_date={} and end_date={}".format(
                start_date.strftime("%Y-%m-%d %H:%M:%S"), 
                end_date.strftime("%Y-%m-%d %H:%M:%S"),
            ))

            # First filter by directory
            # The raw acquisition server creates a new folder for saving raw acquisition
            # data every time fpga_master is restarted. Directories are named:
            # {isotime}_{corr_name}_rawadc, 
            # where isotime is the time & date in ISO format,
            # and corr_name is a descriptive name for the run
            raw_acq_dirs = glob.glob("{}/*_gbo_rawadc*".format(self.raw_acq_dir))
            raw_acq_dirs.sort(key=os.path.getmtime)
            raw_acq_dirs = np.array(raw_acq_dirs)
            raw_acq_start_dates = np.array([dateutil.parser.isoparse(re.split("/|_", d)[-3]) for d in raw_acq_dirs])
            # TODO: Update this to be faster for more recent data? Will be important once we have been operating
            # for a while.
            inds = np.where((raw_acq_start_dates <= end_date))[0] # (raw_acq_start_dates >= start_date) & 
            if inds[0] - 1 >= 0:
                inds = np.concatenate(([inds[0]-1], inds))

            # Then filter by files in those directories
            files = []
            for d in raw_acq_dirs[inds]:
                fs = glob.glob("{}/*h5".format(d))
                files = np.concatenate((files, fs))
            files = files.tolist()
            files.sort(key=os.path.getmtime)
            file_dates = np.array([utc.localize(datetime.datetime.utcfromtimestamp((os.path.getmtime(f)))) for f in files])
            inds = np.where((file_dates >= start_date) & (file_dates <= end_date))[0]
            if len(inds) == 0:
                time_delta = datetime.timedelta(hours=1)
                inds = np.where((file_dates >= start_date-time_delta) & (file_dates <= end_date+time_delta))[0]
            inds = np.concatenate((inds,[inds[-1]+1]))
            inds = np.concatenate(([inds[0]-1],inds))
            filenames = np.array(files)[inds]
        
        # Read in filenames and populate data and metadata
        filenames = np.array(filenames).tolist()
        filenames.sort(key=os.path.getmtime)
        file_dates = np.array([utc.localize(datetime.datetime.utcfromtimestamp((os.path.getmtime(f)))) for f in filenames])
        log.info("Reading in filenames corresponding to the following times:")
        for jj in range(len(filenames)):
            log.info("{} : {}".format(file_dates[jj].strftime("%Y-%m-%d %H:%M:%S"), filenames[jj]))
        
        for ii, fn in enumerate(filenames):
            # Skip any filenames that are still locked (actively being written)
            try:
                f_h5 = h5py.File(fn, 'r')
            except Exception as e:
                if len(filenames) == 1:
                    raise RawAcqException(
                        "Cannot load indicated file, it is locked: {}".format(fn)
                    )
                continue
            
            # TODO: Update this section later so that the proper SMA input mappings are pulled from chimedb
            
            # Load in the metadata indicating which crate, slot, and input each packet comes from
            crate = f_h5["crate"][:,0]
            fpga_slot = f_h5["slot"][:, 0]
            sma_input = f_h5["adc_input"][:, 0]
            
            # Read in the ctime timestamps that indicate when each packet was taken (packets from
            # the same frame will have the same ctime)
            timestamp = f_h5["timestamp"][:, 0]
            ctime = timestamp["ctime"]
            fpga_count = timestamp["fpga_count"]
            
            # Load in the actual timestream of voltage values in LSB = Least Significant Bit units
            # This timestream will have shape: [npackets, 2048], where npackets is all of the frames
            # saved in crate, slot, input, and time space, unraveled.
            timestream_fn = f_h5["timestream"][:]
            # Note that crate, fpga_slot, sma_input, ctime will have len=npackets
            
            f_h5.close()
            
            # Concatenate together timestream and metadata arrays
            if ii == 0:
                crates = crate
                slots = fpga_slot
                inputs = sma_input
                ctimes = ctime
                fpga_counts = fpga_count
                timestream = timestream_fn
            else:
                crates = np.concatenate((crates, crate))
                slots = np.concatenate((slots, fpga_slot))
                inputs = np.concatenate((inputs, sma_input))
                ctimes = np.concatenate((ctimes, ctime))
                fpga_counts = np.concatenate((fpga_counts, fpga_count))
                timestream = np.concatenate((timestream, timestream_fn))
        
        # Complete one more time filter for time at the frame level (30 second resolution)
        frame_datetimes = np.array([pytz.utc.localize(datetime.datetime.fromtimestamp(ctime)) for ctime in ctimes])
        inds = np.where((frame_datetimes >= start_date) & (frame_datetimes <= end_date))[0]
        
        # Save the final arrays in the object
        self.crates = crates[inds]
        self.slots = slots[inds]
        self.inputs = inputs[inds]
        self.ctimes = ctimes[inds]
        self.fpga_counts = fpga_counts[inds]
        self.timestream = timestream[inds]
        
        # Calculate and save some metadata
        # ctime timestamp of first frame in this file
        self.start_time = utc.localize(datetime.datetime.fromtimestamp(self.ctimes[0]))
        # ctime timestamp of last frame in this file
        self.end_time = utc.localize(datetime.datetime.fromtimestamp(self.ctimes[-1]))
        # Figure out the frame time for each unique frame, and, therefore, how many frames were saved
        uniq_fpga_count, iuniq, itime = np.unique(self.fpga_counts, return_index=True, return_inverse=True)
        self.ctime_frames = self.ctimes[iuniq]
        self.num_crates = np.max(self.crates) + 1
        self.num_slots = np.max(self.slots) + 1
        self.num_inputs = np.max(self.inputs) + 1
        self.num_frames = len(self.ctime_frames) + 1
      
    
    def get_timestream_for_input(self, crate_number : int, slot_number : int, input_number : int):
        """ This function selects the voltage timestream for a given crate number, slot number, and input number
    
        Parameters
        ----------
        crate_number : int
            A number indicating what crate to select
            (usually only crate=0 for CHIME Outriggers)
        slot_number : int
            A number indicating which FPGA slot to select (a number 0 to 15)
        input_number : int
            A number indicating which SMA input to select on the FPGA indicated by slot (a number 0 to 15)    
        
        Outputs
        -------
        timestream_input : np.ndarray of int
            An array of the timestream of voltage values in LSB for the given crate, slot, and input
            Will have shape: [nframes, 2048] where nframes is the number of frames for the given input
        ctimes : np.ndarray of float
            An array of metadata indicating the ctime of each frame for the given crate, slot, and input
        fpga_counts : np.ndarray of int
            An array of metadata indicating the fpga timestamp of each frame for the given crate, slot, and input
        """
        # Select the indices in the timestream for the given crate, slot, input
        # Note the "reduce" is necessary because intersect1d only allows two arrays
        inds = reduce(
            np.intersect1d,
            (
                np.where(self.crates == crate_number),
                np.where(self.slots == slot_number),
                np.where(self.inputs == input_number),
            )
        )
        
        ctimes = self.ctimes[inds]
        fpga_counts = self.fpga_counts[inds]
        timestreams = self.timestream[inds]
        
        return timestreams, ctimes, fpga_counts
        
        
    def calc_rms(self, crate_number : int, slot_number : int, input_number : int):
        """ This function calculates the rms of the voltage values for each frame in the 
        dataset corresponding to a given crate number, slot number, and input number
        
        Parameters
        ----------
        crate_number : int
            A number indicating what crate to select
            (usually only crate=0 for CHIME Outriggers)
        slot_number : int
            A number indicating which FPGA slot to select (a number 0 to 15)
        input_number : int
            A number indicating which SMA input to select on the FPGA indicated by slot (a number 0 to 15)    
        
        Outputs
        -------
        rms : np.ndarray of float
            An array of the rms of the voltage values in LSB for the given crate, slot, and input
            Will have len=nframes where nframes is the number of frames for the given input
        """
        timestreams, ctimes, fpga_counts = self.get_timestream_for_input(crate_number, slot_number, input_number)
        # rms = np.sqrt(np.mean(np.square(timestreams), axis=1))
        # Note that we choose to plot the STD instead of the RMS here, as the STD
        # is what we plot on Grafana
        rms = np.nanstd(timestreams, axis=1)
        
        return rms
    
    
    def calc_kurtosis(self, crate_number : int, slot_number : int, input_number : int):
        """ This function calculates the kurtosis of the voltage values for each frame in the 
        dataset corresponding to a given crate number, slot number, and input number
        
        Parameters
        ----------
        crate_number : int
            A number indicating what crate to select
            (usually only crate=0 for CHIME Outriggers)
        slot_number : int
            A number indicating which FPGA slot to select (a number 0 to 15)
        input_number : int
            A number indicating which SMA input to select on the FPGA indicated by slot (a number 0 to 15)    
        
        Outputs
        -------
        kurtosis : np.ndarray of float
            An array of the kurtosis of the voltage values in LSB for the given crate, slot, and input
            Will have len=nframes where nframes is the number of frames for the given input
        """
        timestreams, ctimes, fpga_counts = self.get_timestream_for_input(crate_number, slot_number, input_number)
        kurtosis = scipy_kurtosis(timestreams, axis=1)
        
        return kurtosis
    
    
    def calc_mean(self, crate_number : int, slot_number : int, input_number : int):
        """ This function calculates the mean of the voltage values for each frame in the 
        dataset corresponding to a given crate number, slot number, and input number
        
        Parameters
        ----------
        crate_number : int
            A number indicating what crate to select
            (usually only crate=0 for CHIME Outriggers)
        slot_number : int
            A number indicating which FPGA slot to select (a number 0 to 15)
        input_number : int
            A number indicating which SMA input to select on the FPGA indicated by slot (a number 0 to 15)    
        
        Outputs
        -------
        mean : np.ndarray of float
            An array of the mean of the voltage values in LSB for the given crate, slot, and input
            Will have len=nframes where nframes is the number of frames for the given input
        """
        timestreams, ctimes, fpga_counts = self.get_timestream_for_input(crate_number, slot_number, input_number)
        mean = np.nanmean(timestreams, axis=1)
        
        return mean
    
    
    def calc_fft(self, crate_number : int, slot_number : int, input_number : int):
        """ This function calculates the FFT of the voltage values for each frame in the 
        dataset corresponding to a given crate number, slot number, and input number
        
        Parameters
        ----------
        crate_number : int
            A number indicating what crate to select
            (usually only crate=0 for CHIME Outriggers)
        slot_number : int
            A number indicating which FPGA slot to select (a number 0 to 15)
        input_number : int
            A number indicating which SMA input to select on the FPGA indicated by slot (a number 0 to 15)  
        
        Outputs
        -------
        fft : np.ndarray of float
            An array of the FFT of the voltage values for each frame for the given crate, slot, and input
            Will have shape: [nframes, nfreqs] where nframes is the number of frames for the given input and
            nfreqs is the number of frequency bins in the Fourier transform (2048/2=1024)
        
        avg_fft : np.ndarray of float
            The average FFT across all frames for the given crate, slot, and input
        """
        timestreams, ctimes, fpga_counts = self.get_timestream_for_input(crate_number, slot_number, input_number)

        npackets = timestreams.shape[1]
        fft = np.abs(np.fft.fft(np.hamming(npackets) * (timestreams - np.mean(timestreams, axis=1)[:, np.newaxis]), axis=1))**2
        fft = fft[:,:npackets//2][:,::-1]
        avg_fft = np.median(fft, axis=0)
        
        return fft, avg_fft
    
    
    def get_timeseries(self, crate_number : int, slot_number : int, input_number : int):
        """ This function retrieves a timeseries of the voltage values in LSB units by combining
        all frames corresponding to a given crate number, slot number, and input number together 
        in temporal sequence.
        
        Parameters
        ----------
        crate_number : int
            A number indicating what crate to select
            (usually only crate=0 for CHIME Outriggers)
        slot_number : int
            A number indicating which FPGA slot to select (a number 0 to 15)
        input_number : int
            A number indicating which SMA input to select on the FPGA indicated by slot (a number 0 to 15)  
        
        Outputs
        -------
        timestreams_ravel : np.ndarray of float
            An array of the voltage values for each frame for the given crate, slot, and input
            Will have shape: [nframes, nfreqs] where nframes is the number of frames for the given input and
            nfreqs is the number of frequency bins in the Fourier transform (2048/2=1024)
        """
        timestreams, ctimes, fpga_counts = self.get_timestream_for_input(crate_number, slot_number, input_number)
        timestreams_ravel = timestreams.ravel()
        
        return timestreams_ravel
    
    
    def plot_input_summary_diagnostic(
        self, 
        inputs : np.ndarray = np.array([[0,0,14],[0,2,12],[0,4,10],[0,6,8],[0,8,6],[0,10,4],[0,12,2],[0,14,0]]), 
        plot_types : np.ndarray = ['rms', 'dynspec', 'hist'],
        plot_filename : str = None,
        log_histogram : bool = False, 
        log_spectrum : bool = True, 
        # separate_polarizations : bool = False,
    ):
        """ This function creates a summary plot that provides an overview of the feed data quality
        by plotting diagnostics for the specified inputs side-by-side in one PDF. 
        
        Parameters
        ----------
        inputs : np.ndarray of int
            An Nx3 array specifying N inputs to plot. Each N entry is a coordinate: [crate, slot, input]
        plot_types : np.ndarray of str
            A 1D array of strings indicating which dictate which plots to include in the summary. 
            The strings must be of the following choices:
            plot_types = ['rms', 'timeseries', 'avgfft', 'dynspec', 'hist']
        plot_filename : str
            The name of the plot file, if you want it to be different than default
        log_histogram : bool
            Indicates if plotted histograms should be logarithmic
        log_spectrum : bool
            Indicates if plotted average spectra should be logarithmic
        
        Outputs
        -------
        sample_{start_date}_{end_date}_{plot_types}_raw_acq_summary.pdf
            A pdf of the plot, located in the plot_dir directory.
        """
        log.info("Plotting summary plot with \ninputs={} and \nplot_types={}".format(inputs, plot_types))
        
        # First, verify inputs
        if inputs.shape[1] != 3:
            raise RawAcqException(
                "Input an Nx3 array specifying N inputs to plot, where each entry is a coordinate: [crate, slot, input]."
            )
        for pt in plot_types:
            if pt not in ['rms', 'timeseries', 'avgfft', 'dynspec', 'hist']:
                raise RawAcqException(
                    "Plot types must be of the following choices: ['rms', 'timeseries', 'avgfft', 'dynspec', 'hist']."
                )
        
        if plot_filename is None:
            plot_name = "{}/sample_{}_{}".format(
                self.plot_dir,
                self.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                self.end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            for plot_type in plot_types:
                plot_name = plot_name + "_{}".format(plot_type)
            plot_name = plot_name + ".pdf"
        else:
            plot_name = "{}/{}.pdf".format(
                self.plot_dir,
                plot_name,
            )
        log.info("Outputting plot to: {}".format(plot_name))
        p = PdfPages(plot_name)
        
        # Iterate through inputs
        num_inputs_to_plot = len(inputs)
        num_inputs_per_page = 8
        num_plot_types = len(plot_types)
        ii = 0
        for mm in range(num_inputs_to_plot):
            crate_number = inputs[mm][0]
            slot_number = inputs[mm][1]
            input_number = inputs[mm][2]
            
            if mm % num_inputs_per_page == 0:
                # Set up figure size based on number of inputs and number of plots
                fig = plt.figure(figsize=(8.5, 13))
                figtitle = "ADC"
                for plot_type in plot_types:
                    figtitle = figtitle + " {}".format(plot_type).upper()
                figtitle = figtitle + "\n"
                figtitle = figtitle + "START: {} UTC ".format(self.start_time.strftime("%Y-%m-%dT%H:%M:%S"))
                figtitle = figtitle + "END: {} UTC ".format(self.end_time.strftime("%Y-%m-%dT%H:%M:%S"))
                fig.suptitle(figtitle)
            
            log.info("Plotting crate={}, slot={}, input={}".format(
                crate_number, slot_number, input_number
            ))
            
            timestreams, ctimes, fpga_counts = self.get_timestream_for_input(crate_number, slot_number, input_number)
            
            # Iterate through plot types, plotted left to right
            for jj in range(num_plot_types):
                plot_type = plot_types[jj]
                ax = fig.add_subplot(num_inputs_per_page, num_plot_types, num_plot_types*ii+jj+1)
                
                if plot_type == 'rms':
                    log.info("Plotting rms")
                    rms = self.calc_rms(crate_number, slot_number, input_number)
                    ts = (fpga_counts - fpga_counts[0]) * 2.56e-6 # in seconds
                    ax.plot(ts / 60., rms, lw=0.8, color='black', rasterized=True)
                    ax.grid()
                    ax.set_xlabel('Minutes')
                    ax.set_ylabel("LSB")
                    ax.set_title('RMS')
                    if np.nanmean(rms) > 2:
                        ax.set_ylim([2**(1),2**7])
                    else:
                        ax.set_ylim([2**(-3),2**4])
                    # ax.set_ylim([2**(-2),2**6+1])
                    ax.set_yscale('log', base=2)
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                
                if plot_type == 'timeseries':
                    log.info("Plotting timeseries")
                    timeseries = self.get_timeseries(crate_number, slot_number, input_number)
                    sample_number = np.arange(len(timeseries))
                    
                    num_frames_to_plot = 100
                    num_samples_to_plot = num_frames_to_plot * timestreams.shape[1]
                    timeseries = timeseries[:num_samples_to_plot]
                    sample_number = sample_number[:num_samples_to_plot]
                    
                    ax.plot(sample_number, timeseries, lw=0.8, color='black', rasterized=True)
                    ax.grid()
                    ax.set_xlabel('Samples')
                    ax.set_ylabel('LSB')
                    ax.set_title('Timeseries')
                    ax.set_ylim([-128, 128])
                    ax.set_xlim([0, num_samples_to_plot])
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    
                if plot_type == 'avgfft':
                    log.info("Plotting average FFT")
                    fft, avg_fft = self.calc_fft(crate_number, slot_number, input_number)
                    freqs = np.linspace(400., 800., len(avg_fft))
                    # Note that we throw away the DC bin at 800 MHz for plotting clarity
                    ax.plot(freqs[:-1], avg_fft[:-1], lw=0.8, color='black', rasterized=True)
                    ax.set_xlabel('MHz')
                    ax.set_ylabel('Power')
                    ax.set_title('Avg Spectrum')
                    ax.set_xlim([400, 800])
                    if log_spectrum:
                        ax.set_yscale('log')
                        if np.nanmean(avg_fft) > 10**2:
                            ax.set_ylim([5*10**3, 2*10**6])
                
                if plot_type == 'dynspec':
                    log.info("Plotting dynamic spectrum")
                    fft, avg_fft = self.calc_fft(crate_number, slot_number, input_number)
                    ts = (fpga_counts - fpga_counts[0]) * 2.56e-6 # in seconds
                    # Note that we throw away the DC bin at 800 MHz for plotting clarity
                    ax.imshow(
                        fft,
                        aspect='auto',
                        vmin=np.percentile(fft, 10),
                        vmax=np.percentile(fft, 90),
                        extent=[400, 800, ts[-1]/60., ts[0]/60.],
                        interpolation='None',
                        rasterized=True,
                    )
                    ax.set_xlabel('MHz')
                    ax.set_ylabel('Minutes')
                    ax.set_title('Spectrum')
                    ax.set_xlim([400, 800])
                    
                if plot_type == 'hist':
                    log.info("Plotting histogram")
                    ax.hist(timestreams.ravel(), range=[-128, 127], bins=256, color='black', density=True, rasterized=True)
                    if log_histogram:
                        ax.set_yscale('log')
                    ax.set_ylim(bottom=-0.001)
                    ax.set_xlim([-128, 128])
                    ax.text(0.02, 0.8, '$\\sigma=$%0.1f bits' %(np.log2(np.std(timestreams.ravel()))), transform=ax.transAxes, fontsize=9)
                    ax.set_title("Histogram")
                
                if jj == 0:
                    ax.set_ylabel("FPGA slot: {}\nSMA input: {}".format(slot_number, input_number))
                    
            if ii == num_inputs_per_page-1 or mm == num_inputs_to_plot-1:
                fig.tight_layout(pad=2, w_pad=1, h_pad=1)
                p.savefig(fig)
                plt.close('all')
                ii = 0
            else:
                ii = ii + 1
        p.close()
        
        
    def downsample(self, intensity, time_factor=1, freq_factor=1):
        """ This function mean downsamples the given dynamic spectrum by a factor in time and/or frequency.
            Note that the intensity must have shape: [time,freq]

            Parameters
            ----------
            intensity : np.ndarray of float
                A 2D array representing a dynamic spectrum in [time,freq] shape
            time_factor : int
                An integer indicating the factor to reduce the number of time samples by
            freq_factor : int
                An integer indicating the factor to reduce the number of spectral subbands by

            Outputs
            -------
            intensity : np.ndarray of float
                A 2D array representing a dynamic spectrum downsampled to in [time//time_factor,freq//freq_factor] shape
        """

        if time_factor > 1:
            nsamp = intensity.shape[0]

            new_num_times = int(nsamp / time_factor)
            num_to_trim = nsamp % time_factor

            if num_to_trim > 0:
                intensity = intensity[:-num_to_trim,:]

            intensity = np.array(
                np.row_stack(
                    [
                        np.nanmean(subint, axis=0)
                        for subint in np.vsplit(intensity, new_num_times)
                    ]
                )
            )
        if freq_factor > 1:
            nsamp = intensity.shape[1]

            new_num_spectra = int(nsamp / freq_factor)
            num_to_trim = nsamp % freq_factor

            if num_to_trim > 0:
                intensity = intensity[:,:-num_to_trim]

            intensity = np.array(
                np.column_stack(
                    [
                        np.nanmean(subint, axis=1)
                        for subint in np.hsplit(intensity, new_num_spectra)
                    ]
                )
            )
        return intensity
    
    
    def plot_total_dynamic_spectrum(  
        self,
        mask_rfi : bool = True,
        mask_sun : bool = True,
        ds_time_factor : int = 3,
        ds_freq_factor : int = 1,
        site : str = 'gbo',
        save_plot : bool = True,
        figsize = (8.27,11.69),
        bad_inputs = BAD_INPUTS,
        ):
        """ This function plots the dynamic spectrum over the entire temporal period that 
            has been read into the RawAcq object, summed over all inputs (ignoring bad inputs). 
            An average timeseries and spectrum are also plotted.

            Parameters
            ----------
            mask_rfi : bool
                Indicates if the three most significant persistent RFI channels will be masked
            mask_sun : bool
                Indicates if solar transit will be masked (if present)
            ds_time_factor : int
                An integer indicating the factor to reduce the number of time samples by
            ds_freq_factor : int
                An integer indicating the factor to reduce the number of spectral subbands by
            site : str
                A string indicating the site that the data is being plotted for. Should be one
                of the following: ['gbo', 'chime', 'kko', 'pco', 'hco']
            save_plot : bool
                Indicates whether to save a PDF of the plot in plot_dir
            bad_inputs : list of int
                 An Nx3 list specifying bad inputs to avoid. Each N entry is a coordinate: [crate, slot, input]

            Outputs
            -------
            dynamic_spectrum_{start_time}_{end_time}.pdf
        """    
        log.info("Calculating and stacking FFTs over all inputs (takes a bit)")

        crate_numbers = np.zeros(1)
        input_numbers = np.arange(16)
        slot_numbers = np.arange(16)
        inputs = np.array(list(itertools.product(crate_numbers, slot_numbers, input_numbers)), dtype=int)
        # Keep track of ctimes to help with masking solar transit
        ctimes_all = []
        fft_all = []

        for ii in inputs:
            crate_number = ii[0]
            slot_number = ii[1]
            input_number = ii[2]
            
            for bi in BAD_INPUTS:
                if bi[0] == ii[0] and bi[1] == ii[1] and bi[2] == ii[2]:
                    log.info("Skipping bad input {}".format(ii))
                    continue

            fft, _ = self.calc_fft(crate_number, slot_number, input_number)
            _, ctimes, _ = self.get_timestream_for_input(crate_number, slot_number, input_number)

            fft_all.append(fft)
            ctimes_all.append(ctimes)

        # Calculate the minimum number of frames for each input and even out the array
        # TODO: explore why there are fewer frames for some inputs
        n_frames = []
        for ff in fft_all:
            n_frames.append(ff.shape[0])
        min_n_frames = np.min(n_frames)

        for ii in range(len(fft_all)):
             fft_all[ii] = np.array(fft_all[ii][:min_n_frames])
        for ii in range(len(ctimes_all)):
             ctimes_all[ii] = np.array(ctimes_all[ii][:min_n_frames])

#         log.info("Flagging {} bad inputs: {}".format(len(bad_inputs), bad_inputs))
#         fft_all_flagged = np.delete(fft_all, bad_inputs, axis=0)
        fft_all_flagged = fft_all
        log.info("Sum dynamic spectrum over all inputs")
        fft_all_summed = np.sum(fft_all_flagged, axis=0)
        ctimes_all_averaged = np.array([pytz.utc.localize(datetime.datetime.fromtimestamp(ctime)) for ctime in np.mean(ctimes_all, axis=0)])
        start_time = self.start_time
        end_time = self.end_time

        # Note: will be the same for determining transit time regardless of site, 
        # since outriggers are pointed at the same FOV
        site_coords = ephem.Observer()
        site_coords.lat = np.deg2rad(constants.CHIME_LATITUDE_DEG)
        site_coords.long = np.deg2rad(constants.CHIME_LONGITUDE_DEG)
        if site == 'gbo':
            tz = pytz.timezone('US/Eastern')
            tz_dateutil = dateutil.tz.gettz('US/Eastern')
        elif site in ['chime', 'kko', 'pco', 'hco']:
            tz = pytz.timezone('Canada/Pacific')
            tz_dateutil = dateutil.tz.gettz('Canada/Pacific')
        else:
            raise RawAcqException(
                "Site must be of the following choices: ['gbo', 'chime', 'kko', 'pco', 'hco']."
            )

        start_time_local = start_time.astimezone(tz)
        end_time_local = end_time.astimezone(tz)
        start_time_mpl = mdates.date2num(start_time_local)
        end_time_mpl = mdates.date2num(end_time_local)

        fig = plt.figure(figsize=figsize)
        
        # (left, bottom, width, height)
        im = (0.0, 0, 0.7, 0.7)
        im_ts = (0.7 + 0.01, 0, 0.15, 0.7)
        im_spec = (0, 0.7, 0.7, 0.15)

        ax = plt.axes(im)

        plot_intensity = fft_all_summed.copy()

        if mask_rfi:
            log.info("Masking the three most significant RFI channels")
            # Cut off the three major TV RFI channels, empirically determined
            rfi_channels = [
                378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393,
                424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
                470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485,
            ]# np.where(avg_spec_all > 0.5e8)[0]
            plot_intensity[:,rfi_channels] = np.nan
        if mask_sun:
            log.info("Masking solar transit, if present")
            # Calculate when the Sun transits
            sun = ephem.Sun()
            transit_time = pytz.utc.localize(site_coords.next_transit(sun, start_time).datetime())#.astimezone(tz)
            # Select 5 minutes on either side
            time_delta = datetime.timedelta(minutes=15)
            start_sun_transit = transit_time - time_delta
            end_sun_transit = transit_time + time_delta
            sun_channels = np.where((ctimes_all_averaged >= start_sun_transit) & (ctimes_all_averaged <= end_sun_transit))[0]
            plot_intensity[sun_channels,:] = np.nan

        if ds_time_factor > 1 or ds_freq_factor > 1:
            log.info("Downsampling by factors of: time_factor={0} ({1:.2f} mins), freq_factor={2} ({3:.2f} MHz)".format(
                ds_time_factor, (ds_time_factor * 30. / 60.),
                ds_freq_factor, (ds_freq_factor * 400. / 1024.),
            ))
            plot_intensity = self.downsample(plot_intensity, time_factor=ds_time_factor, freq_factor=ds_freq_factor)

        avg_ts_all = np.nanmean(plot_intensity, axis=1)
        avg_spec_all = np.nanmean(plot_intensity, axis=0)

        ax.imshow(
            plot_intensity,
            aspect='auto',
            vmin=np.nanpercentile(plot_intensity, 5),
            vmax=np.nanpercentile(plot_intensity, 95),
            extent=[400, 800, end_time_mpl, start_time_mpl],
            interpolation='None',
            rasterized=True,
        )
        date_format = mdates.DateFormatter('%H:%M', tz=tz_dateutil) # :%S
        ax.yaxis.set_major_formatter(date_format)
        ax.yaxis_date(tz=tz)
        ax.set_ylabel("Time")
        ax.set_xlabel("Frequency (MHz)")

        ax = plt.axes(im_ts)

        ax.plot(avg_ts_all[::-1], range(len(avg_ts_all)), lw=0.8, color='black', rasterized=True)
        ax.set_ylim([0,len(avg_ts_all[::-1])])
        ax.get_yaxis().set_ticks([])

        ax = plt.axes(im_spec)

        ax.plot(np.arange(len(avg_spec_all))[:-1], avg_spec_all[:-1], lw=0.8, color='black', rasterized=True)
        ax.set_xlim([0,len(avg_spec_all[::-1])])
        ax.set_yscale('log')
        ax.get_xaxis().set_ticks([])

        ax.set_title("{} Raw Acq\n{} to {}".format(
            site.upper(),
            start_time_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            end_time_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        ))

        plot_name = "{}/dynamic_spectrum_{}_{}.pdf".format(
            self.plot_dir,
            self.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            self.end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        if save_plot:
            log.info("Saving {}".format(plot_name))
            plt.savefig(
                plot_name,
                dpi=300,
                bbox_inches="tight",
            )
        #plt.close()

    def plot_slot_dynamic_spectrum_summary(  
        self,
        crate_number : int,
        slot_number : int,
        mask_rfi : bool = True,
        mask_sun : bool = True,
        ds_time_factor : int = 3,
        ds_freq_factor : int = 1,
        site : str = 'gbo',
        plot_filename : str = None,
        save_plot : bool = True,
        figsize = (7,8),
        ):
        """ This function plots the dynamic spectra over the entire temporal period that 
            has been read into the RawAcq object, for each individual input in a given slot (there
            should be 16 dynamic spectra per PDF). An average timeseries and spectrum are also plotted.

            Parameters
            ----------
            crate_number : int
                A number indicating what crate to select
                (usually only crate=0 for CHIME Outriggers)
            slot_number : int
                A number indicating which FPGA slot to select (a number 0 to 15)
            mask_rfi : bool
                Indicates if the three most significant persistent RFI channels will be masked
            mask_sun : bool
                Indicates if solar transit will be masked (if present)
            ds_time_factor : int
                An integer indicating the factor to reduce the number of time samples by
            ds_freq_factor : int
                An integer indicating the factor to reduce the number of spectral subbands by
            site : str
                A string indicating the site that the data is being plotted for. Should be one
                of the following: ['gbo', 'chime', 'kko', 'pco', 'hco']
            plot_filename : str
                The name of the plot file, if you want it to be different than default
            save_plot : bool
                Indicates whether to save a PDF of the plot in plot_dir

            Outputs
            -------
            dynamic_spectra_crate{crate_number}_slot{slot_number}_{start_time}_{end_time}.pdf
        """  
        
        if plot_filename is None:
            plot_name = "{}/dynamic_spectra_crate{}_slot{}_{}_{}.pdf".format(
                self.plot_dir,
                crate_number,
                slot_number,
                self.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                self.end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
        else:
            plot_name = "{}/{}.pdf".format(
                self.plot_dir,
                plot_name,
            )
        log.info("Outputting plot to: {}".format(plot_name))
        p = PdfPages(plot_name)
        
        num_inputs_to_plot = self.num_inputs
        for input_number in range(num_inputs_to_plot):
            
            log.info("Plotting crate={}, slot={}, input={}".format(
                crate_number, slot_number, input_number
            ))
            
            # Set up figure
            fig = plt.figure(figsize=figsize)
            
            log.info("Calculating FFT for input {}".format(input_number))
            fft, _ = self.calc_fft(crate_number, slot_number, input_number)
            _, ctimes, _ = self.get_timestream_for_input(crate_number, slot_number, input_number)
            
            ctimes_dates = np.array([pytz.utc.localize(datetime.datetime.fromtimestamp(ctime)) for ctime in ctimes])
            start_time = self.start_time
            end_time = self.end_time
            # Note: will be the same for determining transit time regardless of site, 
            # since outriggers are pointed at the same FOV
            site_coords = ephem.Observer()
            site_coords.lat = np.deg2rad(constants.CHIME_LATITUDE_DEG)
            site_coords.long = np.deg2rad(constants.CHIME_LONGITUDE_DEG)
            if site == 'gbo':
                tz = pytz.timezone('US/Eastern')
                tz_dateutil = dateutil.tz.gettz('US/Eastern')
            elif site in ['chime', 'kko', 'pco', 'hco']:
                tz = pytz.timezone('Canada/Pacific')
                tz_dateutil = dateutil.tz.gettz('Canada/Pacific')
            else:
                raise RawAcqException(
                    "Site must be of the following choices: ['gbo', 'chime', 'kko', 'pco', 'hco']."
                )
            log.info("Converting timestamps to local time: {}".format(tz))
            start_time_local = start_time.astimezone(tz)
            end_time_local = end_time.astimezone(tz)
            start_time_mpl = mdates.date2num(start_time_local)
            end_time_mpl = mdates.date2num(end_time_local)
            
            # (left, bottom, width, height)
            im = (0.13, 0.08, 0.7, 0.7)
            im_ts = (0.7 + 0.13, 0.08, 0.15, 0.7)
            im_spec = (0.13, 0.8-0.02, 0.7, 0.15)

            ax = plt.axes(im)

            plot_intensity = fft.copy()

            if mask_rfi:
                log.info("Masking the three most significant RFI channels")
                # Cut off the three major TV RFI channels, empirically determined
                rfi_channels = [
                    378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393,
                    424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
                    470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485,
                ]# np.where(avg_spec_all > 0.5e8)[0]
                plot_intensity[:,rfi_channels] = np.nan
            if mask_sun:
                log.info("Masking solar transit, if present")
                # Calculate when the Sun transits
                sun = ephem.Sun()
                transit_time = pytz.utc.localize(site_coords.next_transit(sun, start_time).datetime())#.astimezone(tz)
                # Select 5 minutes on either side
                time_delta = datetime.timedelta(minutes=15)
                start_sun_transit = transit_time - time_delta
                end_sun_transit = transit_time + time_delta
                sun_channels = np.where((ctimes_dates >= start_sun_transit) & (ctimes_dates <= end_sun_transit))[0]
                plot_intensity[sun_channels,:] = np.nan

            if ds_time_factor > 1 or ds_freq_factor > 1:
                log.info("Downsampling by factors of: time_factor={0} ({1:.2f} mins), freq_factor={2} ({3:.2f} MHz)".format(
                    ds_time_factor, (ds_time_factor * 30. / 60.),
                    ds_freq_factor, (ds_freq_factor * 400. / 1024.),
                ))
                plot_intensity = self.downsample(plot_intensity, time_factor=ds_time_factor, freq_factor=ds_freq_factor)

            avg_ts = np.nanmean(plot_intensity, axis=1)
            avg_spec = np.nanmean(plot_intensity, axis=0)

            vmin = np.nanpercentile(plot_intensity, 10)
            vmax = np.nanpercentile(plot_intensity, 90)
            ax.imshow(
                plot_intensity,
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                extent=[400, 800, end_time_mpl, start_time_mpl],
                interpolation='None',
                rasterized=True,
            )
            date_format = mdates.DateFormatter('%H:%M', tz=tz_dateutil) # :%S
            ax.yaxis.set_major_formatter(date_format)
            ax.yaxis_date(tz=tz)
            ax.set_ylabel("Time")
            ax.set_xlabel("Frequency (MHz)")

            ax = plt.axes(im_ts)

            ax.plot(avg_ts[::-1], range(len(avg_ts)), lw=0.8, color='black', rasterized=True)
            ax.set_ylim([0,len(avg_ts[::-1])])
            ax.get_yaxis().set_ticks([])

            ax = plt.axes(im_spec)

            ax.plot(np.arange(len(avg_spec))[:-1], avg_spec[:-1], lw=0.8, color='black', rasterized=True)
            ax.set_xlim([0,len(avg_spec[::-1])])
            ax.set_yscale('log')
            ax.get_xaxis().set_ticks([])
            figtitle = "{} Raw Acq, Crate {}, Slot {}, Input {}\n{} to {}".format(
                site.upper(),
                crate_number,
                slot_number,
                input_number,
                start_time_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
                end_time_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            )
            # fig.suptitle(figtitle, y=1.1)
            ax.set_title(figtitle)
        
            log.info("Save page {}".format(input_number))
            plt.tight_layout()
            #fig.tight_layout(pad=2)#, w_pad=1, h_pad=1)
            p.savefig(fig)
            plt.close('all')
        log.info("Save final PDF")
        p.close()
        
#     plot_all_inputs_diagnostic(bools indicating which to plot: rms, timeseries, avgfft, dyn_spec, hist, histlog, speclog, sep_pol)
#     plot_readout_diagnostic(plot showing the readout of samples as a function of input, slot, ctime, etc)
 
        

def identify_gbo_maintenance_days(
    start_date_et, 
    end_date_et,
):
    """ This function queries the GBO observing schedule to identify maintenance days 
        and non-maintenance days within the time period specified by start_time_et and end_time_et.

        Outputs
        -------
        start_time_et : datetime
            The starting time in ET
        end_time_et : datetime
            The ending time in ET

        Outputs
        -------
        maintenance_days : list of str
            A list of date strings in %Y-%m-%d format for the maintenance days in ET
        non_maintenance_days : list of str
            A list of date strings in %Y-%m-%d format for the non-maintenance days in ET
    """ 
    date_diff = end_date_et - start_date_et

    # Define URL to query the GBT schedule for
    url = 'https://dss.gb.nrao.edu/schedule/public/printerFriendly?tz=ET&start={0:02d}/{1:02d}/{2:04d}&days={3}'.format(
        start_date_et.month,
        start_date_et.day,
        start_date_et.year,
        int(date_diff.days)+2,
    )
    r = requests.get(url)
    schedule_str = r.text
    soup = BeautifulSoup(schedule_str)

    date_title_str = "Start and End times for the start day (timezone). '+' indicates that the period continues on in an undisplayed date."
    type_str = "project_type type_maintenance"

    maintenance_days = []
    non_maintenance_days = []
    tablerow = soup.find_all('tr')[0]
    is_maintenance = False
    ii = 0
    # Please excuse the hacky HTML parsing :-)
    for tablerow in soup.find_all('tr'):
        try:
            rowheader = tablerow['class'][0] 
        except:
            rowheader = ''

        if ii == 0:
            date_str = tablerow.find("th", attrs={"title":date_title_str}).text.split()[0]
            ii = ii + 1
        elif rowheader == 'day_header':
            if is_maintenance:
                log.info("{} is a maintenance day".format(date_str))
                maintenance_days.append(date_str)
            else:
                # log.info("{} is not maintenance day".format(date_str))
                non_maintenance_days.append(date_str)

            date_str = tablerow.find("th", attrs={"title":date_title_str}).text.split()[0]
            is_maintenance = False
            ii = ii + 1
        else:
            try:
                project_type = tablerow.find('td', attrs={"class": type_str}).text
                if project_type == "M":
                    is_maintenance = True
            except:
                continue

    return maintenance_days, non_maintenance_days

def mad(data, median=None, ax=0):
    if not median:
        median = np.nanmedian(data, axis=ax)
    return np.nanmedian(np.abs(np.subtract(data, median[..., np.newaxis])), axis=ax)

def plot_maintenance_vs_nonmaintenance_timeseries(
    start_date_et_str : str, 
    end_date_et_str : str, 
    plot_types = ['rms', 'mean', 'mad', 'kurtosis'],
    plot_filename = None,
    bad_inputs = BAD_INPUTS,
    figsize = (11,5),
    plot_dir = './'
):
    """ This function creates plots showing a band-averaged and input-averaged 24 hr timeseries of 
        the raw acq data, with two panels splitting between maintenance and non-maintenance days. 
        The function will produce a separate plot for each of the plot_types, where the plot type 
        indicates whether the timeseries shows the rms, standard deviation, mean, 
        mean absolute deviation, or kurtosis vs time.
        
        Note: this function is still a little janky, and takes a long time to run if you put in 
        a long date range. Be patient.
        
        Parameters
        ----------
        start_date_et_str : str
            A string in format %Y-%m-%d %H:%M:%S indicating the start date in ET that will be plotted
        end_date_et_str : str
            A string in format %Y-%m-%d %H:%M:%S indicating the end date in ET that will be plotted
        plot_types : np.ndarray of str
            A 1D array of strings indicating which dictate which plots to make. 
            The strings must be of the following choices:
            plot_types = ['rms', 'mean', 'mad', 'kurtosis']
        plot_filename : str
            The name of the plot file, if you want it to be different than default
        bad_inputs : list of int
            An Nx3 list specifying bad inputs to avoid. Each N entry is a coordinate: [crate, slot, input]
        
        Outputs
        -------
        maintenance_comparison_timeseries_{start_date}_{end_date}_{plot_types}.pdf
            A pdf of the plot, located in the plot_dir directory.
        """
    for pt in plot_types:
        if pt not in ['rms', 'mean', 'mad', 'kurtosis']:
            raise RawAcqException(
                "Plot types must be of the following choices: ['rms', 'mean', 'mad', 'kurtosis']."
            )
    
    est = pytz.timezone('US/Eastern') #{"EST" : dateutil.tz.gettz('US/Eastern')}
    utc = pytz.utc
    tz = pytz.timezone('US/Eastern')
    tz_dateutil = dateutil.tz.gettz('US/Eastern')

    start_date_et = est.localize(dateutil.parser.parse(start_date_et_str))
    end_date_et = est.localize(dateutil.parser.parse(end_date_et_str))
    start_date_utc = start_date_et.astimezone(utc)
    end_date_utc = end_date_et.astimezone(utc)
    date_range = np.arange(start_date_utc, end_date_utc, datetime.timedelta(days=1))
    
    if plot_filename is None:
        plot_name = "{}/maintenance_comparison_timeseries_{}_{}".format(
            plot_dir,
            start_date_et.strftime("%Y-%m-%dT%H:%M:%S"),
            end_date_et.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        for plot_type in plot_types:
            plot_name = plot_name + "_{}".format(plot_type)
        plot_name = plot_name + ".pdf"
    else:
        plot_name = "{}/{}.pdf".format(
            plot_dir,
            plot_name,
        )

    log.info("Flagging bad inputs")
    crate_numbers = np.zeros(1)
    input_numbers = np.arange(16)
    slot_numbers = np.arange(16)
    inputs = np.array(list(itertools.product(crate_numbers, slot_numbers, input_numbers)), dtype=int)
    inputs_flagged = []
    for ii in inputs:
        bad_input = False
        for bi in BAD_INPUTS:
            if bi[0] == ii[0] and bi[1] == ii[1] and bi[2] == ii[2]:
                log.info("Flagging bad input {}".format(ii))
                bad_input = True
        if not bad_input:
            inputs_flagged.append(ii)
    inputs = np.array(inputs_flagged)
    
    # This dictionary will store results for each plot type
    avg_ts_days = {}
    log.info("Iterate through, gathering the data for all requested plot types...")
    for plot_type in plot_types:
        avg_ts_days[plot_type] = []
        
    date = start_date_utc
    for jj in range(len(date_range)):
        start_date_utc = date
        end_date_utc = date + datetime.timedelta(days=1)

        dates = np.array([
            start_date_utc,
            end_date_utc,
        ])
        raw_acq = RawAcq(dates=dates)

        log.info("Calculating {} for all inputs on dates {} to {}".format(plot_types, dates[0].strftime("%Y-%m-%d %H:%M:%S"), dates[1].strftime("%Y-%m-%d %H:%M:%S")))
        ts_inputs = {}
        for plot_type in plot_types:
            ts_inputs[plot_type] = []
        for ii in inputs:
            crate_number = ii[0]
            slot_number = ii[1]
            input_number = ii[2]
            log.info("Calculating input: {}".format(ii))

            timestreams, ctimes, fpga_counts = raw_acq.get_timestream_for_input(crate_number, slot_number, input_number)

            # ['rms', 'mean', 'mad', 'kurtosis']
            for plot_type in plot_types:
                if plot_type == 'rms':
                    ts_metric = np.nanstd(timestreams, axis=1)
                    ts_inputs[plot_type].append(ts_metric)
                if plot_type == 'mean':
                    ts_metric = np.nanmean(timestreams, axis=1)
                    ts_inputs[plot_type].append(ts_metric)
                if plot_type == 'mad':
                    ts_metric = mad(timestreams, median=None, ax=1)
                    ts_inputs[plot_type].append(ts_metric)
                if plot_type == 'kurtosis':
                    ts_metric = scipy_kurtosis(timestreams, axis=1)
                    ts_inputs[plot_type].append(ts_metric)
        for plot_type in plot_types:
            n_frames = []
            for rr in ts_inputs[plot_type]:
                n_frames.append(rr.shape[0])
            min_n_frames = np.min(n_frames)

            for ii in range(len(ts_inputs[plot_type])):
                 ts_inputs[plot_type][ii] = np.array(ts_inputs[plot_type][ii][:min_n_frames])

            avg_ts_days[plot_type].append(np.nanmean(ts_inputs[plot_type], axis=0))
        date = date + datetime.timedelta(days=1)

    for plot_type in plot_types:
        n_frames = []
        for rr in avg_ts_days[plot_type]:
            n_frames.append(rr.shape[0])
        min_n_frames = np.min(n_frames)

        for ii in range(len(avg_ts_days[plot_type])):
             avg_ts_days[plot_type][ii] = np.array(avg_ts_days[plot_type][ii][:min_n_frames])
        avg_ts_days[plot_type] = np.array(avg_ts_days[plot_type])
        log.info("avg_ts_days[plot_type].shape = {}".format(avg_ts_days[plot_type].shape))
        
    log.info("Identify maintenance vs non-maintenance days between {} and {}".format(start_date_et.strftime("%Y-%m-%d %H:%M:%S"), end_date_et.strftime("%Y-%m-%d %H:%M:%S")))
    maintenance_days, non_maintenance_days = identify_gbo_maintenance_days(
        start_date_et, 
        end_date_et,
    )
        
    log.info("Now creating plots")
    log.info("Outputting plot to: {}".format(plot_name))
    p = PdfPages(plot_name)
    for plot_type in plot_types:
        log.info("Plotting {}".format(plot_type))
        fig = plt.figure(figsize=figsize)
        # (left, bottom, width, height)
        maintenance = (0.0, 0, 0.8, 0.5)
        non_maintenance = (0, 0.5, 0.8, 0.5)

        log.info("Calculate matplotlib timestamps for given plot_type dataset")
        start_date_et_ts = start_date_et
        end_date_et_ts = start_date_et + datetime.timedelta(seconds=avg_ts_days[plot_type].shape[1]*30.)
        incr = (end_date_et_ts - start_date_et_ts) / avg_ts_days[plot_type].shape[1]

        times = [start_date_et_ts]
        for jj in range(avg_ts_days[plot_type].shape[1]-1):
            new_date = times[jj] + incr
            times.append(new_date)
        times = np.array(times)
        times_mpl = mdates.date2num(times)

        roll = 0# -500+67#-600+67

        if len(maintenance_days) > 0:
            ax = plt.axes(maintenance)
            label = False
            date = start_date_et
            for ii in range(len(avg_ts_days[plot_type])):
                color = None

                rms = avg_ts_days[plot_type][ii]
                rms = np.roll(rms,roll)

                date_str = date.strftime("%Y-%m-%d")
                if date_str in maintenance_days:
                    color = 'red'#'xkcd:lighter green'
                    alpha=0.5
                    zorder=10
            #     elif date_str in t_days:
            #         color = 'xkcd:sky'
            #         alpha=1.
            #         zorder=10
                else:
                    color = 'black'
                    alpha=0.2
                    zorder=5

                lw = 2
                if date_str in maintenance_days:
                    if date_str == maintenance_days[0]:
                        ax.plot(times_mpl, rms, lw=lw, color=color, alpha=alpha, zorder=zorder, label="Maintenance Day")
                    else:
                        ax.plot(times_mpl, rms, lw=lw, color=color, alpha=alpha, zorder=zorder)
                    date_format = mdates.DateFormatter('%H:%M', tz=tz_dateutil) # :%S
                    ax.xaxis.set_major_formatter(date_format)
                    ax.xaxis_date(tz=tz)
                    ax.set_xlabel('Local Time')
                    ax.set_ylabel("LSB")
                    ax.legend()
                    # ax.set_ylim([2**(3),35])
                    ax.set_xlim([np.min(times_mpl), np.max(times_mpl)])
                    ax.grid()
                date = date + datetime.timedelta(days=1)

        if len(non_maintenance_days) > 0:
            ax = plt.axes(non_maintenance)
            label = False
            date = start_date_et
            for ii in range(len(avg_ts_days)):
                color = None

                rms = avg_ts_days[plot_type][ii]
                rms = np.roll(rms,roll)

                date_str = date.strftime("%Y-%m-%d")
                if date_str in maintenance_days:
                    color = 'red'#'xkcd:lighter green'
                    alpha=0.5
                    zorder=10
            #     elif date_str in t_days:
            #         color = 'xkcd:sky'
            #         alpha=1.
            #         zorder=10
                else:
                    color = 'black'
                    alpha=0.2
                    zorder=5

                lw = 2
                if date_str not in maintenance_days:
                    if date_str not in maintenance_days and not label:
                        ax.plot(times_mpl, rms, lw=lw, color=color, alpha=alpha, zorder=zorder, label="Non-Maintenance Day")
                        label=True
                    else:
                        ax.plot(times_mpl, rms, lw=lw, color=color, alpha=alpha, zorder=zorder)
                    if len(rms[rms > 40]) > 0:
                        print(date_str)
                    date_format = mdates.DateFormatter('%H:%M', tz=tz_dateutil) # :%S
                    ax.xaxis.set_major_formatter(date_format)
                    ax.xaxis_date(tz=tz)
                    ax.get_xaxis().set_ticklabels([])
                    ax.set_title('{}'.format(plot_type))
                    ax.set_ylabel("LSB")
                    ax.legend()
                    #ax.set_ylim([2**(3),35])
                    ax.set_xlim([np.min(times_mpl), np.max(times_mpl)])
                    ax.grid()
                date = date + datetime.timedelta(days=1)
        p.savefig(fig)
        plt.close('all')
    log.info("Save final PDF")
    p.close()

##############################################################################################################
##############################################################################################################
##############################################################################################################

def main():
    est = pytz.timezone('US/Eastern') #{"EST" : dateutil.tz.gettz('US/Eastern')}
    utc = pytz.utc
    
    # Set the time period that you would like to plot
    start_date_et = est.localize(dateutil.parser.parse("2023-10-15 00:00:00"))
    end_date_et = est.localize(dateutil.parser.parse("2023-10-16 00:00:00"))
    start_date_utc = start_date_et.astimezone(utc)
    end_date_utc = end_date_et.astimezone(utc)
    dates = np.array([
        start_date_utc,
        end_date_utc,
    ])

    # Read in the data
    raw_acq = RawAcq(dates=dates)

    # Make PDFs showing individual dynamic spectra for all inputs for the given time period
    for slot in range(16):
        raw_acq.plot_slot_dynamic_spectrum_summary(crate_number, slot, mask_rfi=False, mask_sun=False, ds_time_factor=1, ds_freq_factor=1, save_plot=True)

    # Make plot of dynamic spectrum summed over all inputs for the given time period
    raw_acq.plot_total_dynamic_spectrum(ds_time_factor=1, ds_freq_factor=1, figsize=(7,7), save_plot=True)

    # Plot various diagnostics like the rms, dynamic spectrum, and histogram of values side-by-side
    # for the given inputs (might be better to run for a shorter time window of data)
    raw_acq.plot_input_summary_diagnostic(
        inputs = np.array([[0,0,14],[0,2,12],[0,4,10],[0,6,8],[0,8,6],[0,10,4],[0,12,2],[0,14,0]]), 
        plot_types= ['rms', 'dynspec', 'hist'],
        plot_filename = None, # Will make a default filename
        log_histogram = False, 
        log_spectrum = True, 
    )

if __name__ == "__main__":
    main()
