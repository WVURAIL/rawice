##################################################################################################
# raw_adc stuff
# iceboard adc Vpp = 0.5, and total 256 levels
# by pranav = yeet
##################################################################################################
import numpy as np
import h5py
import matplotlib.pylab as plt
import datetime
import glob
import os.path
from scipy.signal import get_window
import allantools as allan 
import sys


def progressbar(it, prefix="", size=60, out=sys.stdout):
    '''
    
    Inputs: an array (length of array = number of acq files), a string (displayed at beginning of each line,
            a number (representing the width of the progress bar)
    Outputs: a progress bar denoted by a certain number of '#' corresponding to the percentage of files 
            that have been read
            
        This function is currently only called within the analyse_maser() class to give a visual representation
        of how many of the given files have been analysed.
        
    Example input: 
    array = [0, 1, 2]
    progressbar(array, "Computing Delay: ", 30)
    
    Example output:
    C: path\file0
    C: path\file1
    C: path\file2
    Loaded raw acq HDF5 file... .............................. 0/3
    Checking input ...
    Loaded raw acq HDF5 file... ##########.................... 1/3
    Checking input ...
    Loaded raw acq HDF5 file... ####################.......... 2/3
    Checking input ...
    Done computing delays: ################################### 3/3
    
    DONE reading files and getting delays
    
    A possible error:
    Getting a 'divide by zero' error within the progressbar() function when calling analyse_maser()
        Make sure that the raw acq folder variable has a '/' at the end. 
        raw_acq_folder = "home/users/path/acq" will yield this error
        raw_acq_folder = "home/users/path/acq/" will not yield this error
        
    '''
    count = len(it)
    def show(j):
        x = int(size*j/count) ### dividing by 0 here
        print("{}[{}{}] {}/{}".format(prefix, u"#"*x, "."*(size-x), j, count), end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print(f"Done {prefix}\n", flush=True, file=out)
    
class raw_acq:
    '''
    
    Inputs: a string (path to a single acq file)
    Outputs: "Loaded raw acq file ... "
    
        Assigns the information in the HDF5 acq file to the object initialized with this class. A series of functions will 
        calculate values and assign them to this object.
    
    '''
    def __init__(self, raw_acq_file, diagnostics = False):
        '''
        
        Inputs: a string (path to a single acq file), boolean initialized to false
        Outputs: n/a
        
        The path to the raw acq file is automatically passed to this function, which will called the read() function and 
        and the diagnostics() function if the boolean is set to true. The first argument, self, will tell all proceeding 
        functions to save information to the object defined by the class raw_acq.
        
        '''
        self.file = raw_acq_file
        self.read()
        if diagnostics: 
            self.diagostics()
        
    def read(self):
        '''
        
        Inputs: n/a
        Outputs: "Loaded raw acq HDF5 file ... "
        
        This function is what actually reads in the acq data and saves it to the object. The following values 
        are defined: timestream, timestamp, crate, slot, adc_input, start_time, end_time
        
        '''
        self.hdf5 = h5py.File(self.file,"r")
        index_map = self.hdf5['index_map']
        im_timestream = index_map['timestream'][:]
        #im_snapshot = index_map['snapshot'][:]
        adc_input = np.hstack(self.hdf5['adc_input'][:])
        crate = np.hstack(self.hdf5['crate'][:])
        slot = np.hstack(self.hdf5['slot'][:])
        timestamp = np.hstack(self.hdf5['timestamp'][:])
        timestream = self.hdf5['timestream'][:]
        adc_stream_len = timestream.shape[-1]

        fpga_counts = np.hstack(timestamp['fpga_count'])
        ctime = np.hstack(timestamp['ctime'])
        start_time = datetime.datetime.fromtimestamp(ctime[0]).isoformat()
        end_time = datetime.datetime.fromtimestamp(ctime[-1]).isoformat()

        adc_record_fpga_count_index = np.where(np.roll(fpga_counts,1)!=fpga_counts)[0]
        adc_record_ctime_index = np.where(np.roll(ctime,1)!=ctime)[0]
        adc_record_fpga_count = fpga_counts[adc_record_fpga_count_index]
        adc_record_ctime = fpga_counts[adc_record_ctime_index]
        self.fpga_counts_between_raw_adc_capture = np.diff(adc_record_fpga_count)
        self.time_between_adc_capture = np.unique(self.fpga_counts_between_raw_adc_capture*2.56e-6)
        

        self.num_inputs = np.max(adc_input) + 1
        self.num_crates = np.max(crate) + 1
        self.num_slots = np.max(slot) + 1
        self.num_timestamps = adc_record_fpga_count.shape[0] + 1
        raw_acq.timestream = timestream.astype(int)
        raw_acq.timestamp = timestamp
        raw_acq.crate = crate
        raw_acq.slot = slot
        raw_acq.adc_input = adc_input
        raw_acq.start_time = start_time
        raw_acq.end_time = end_time
        print("Loaded raw acq HDF5 file ... \r")
       
    def diagostics(self):
        '''
        
        Input: n/a
        Output: A list of information about the acq data acquisition as well as a graph that displays the time between
        adc captures. 
        
        You can call this by either calling the object.diagnostics(), or by setting the last argument of raw_acq() to true. 
        
        '''
        print("raw ACQ diagnostics ... \n")
        print(f"archive_version: {self.hdf5.attrs['archive_version'].decode()}")
        print(f"collection_server: {self.hdf5.attrs['collection_server'].decode()}")
        print(f"git_version_tag: {self.hdf5.attrs['git_version_tag'].decode()}")
        print(f"file_name: {self.hdf5.attrs['file_name']}")
        print(f"data_type: {self.hdf5.attrs['data_type'].decode()}")
        print(f"system_user: {self.hdf5.attrs['system_user'].decode()}")
        print(f"rawadc_version: {self.hdf5.attrs['rawadc_version']}")
        print(f"Timestamping_warning: {self.hdf5.attrs['timestamping_warning'].decode()}")
        print()

        print(f"ctime Timestamp of first raw_adc frame: {raw_acq.start_time}")
        print(f"ctime Timestamp of last raw_adc frame: {raw_acq.end_time}")
        print()
        
        plt.figure(figsize=(15,3))
        print(f"Time between raw_adc captures is either {self.time_between_adc_capture} seconds")
        number_adc_captures_to_plot = 150
        plt.scatter(np.arange(number_adc_captures_to_plot)+1,self.fpga_counts_between_raw_adc_capture[:number_adc_captures_to_plot]*2.56e-6)
        plt.ylabel("time since last capture (s)")
        plt.xlabel("rawadc capture number (first capture is #0)")
        plt.title("Time since last adc capture")
        
        
    class check_input:
        '''
        
            Input: a single array corresponding to the input on the ICE board [crate number, slot number, input number]
                Note: input number should be between 0 and 15 (A1 and B8)
            Output: "Checking input [crate number, slot number, input number]..."
            
            This is a sub-class, so the raw_acq object must already exist in order to define this one. It will define a new object
            that holds information for a single input. 
            
        '''
        def __init__(single_inp, input_to_check):
            '''
            
            Input: a single array corresponding to the input on the ICE board [crate number, slot number, input number]
            Output: n/a
            
            Initializes the object and calls a series of functions to calculate and save values. This runs automatically when you run 
            check_input()
            
            '''
            print(f"Checking input {input_to_check} ... \r")
            single_inp.input_to_check = input_to_check
            single_inp.get_timestream_for_input()
            single_inp.get_single_input_rms()
            single_inp.get_fft_of_adc_counts()
            single_inp.get_fgpa_count_for_input()
        
        def get_timestream_for_input(single_inp):
            '''
            
            Input: an array (passed from init())
            Outputs: n/a
            
            Defines the value 'input id' which holds the information for the crate and slot number of the ICE Board as well as the
            input number. Also save the values of time_stamps, which hold the times for each snapshot. Lastly, defines the time streams 
            which are quantized sine waves for the input. 
            
            This function is automatically called when the check_input() object gets initialized.
            
            '''
            itc = single_inp.input_to_check
            input_number = itc[2]
            crate_number = itc[1]
            slot_number = itc[0]
            
            single_inp.time_stamps = raw_acq.timestamp[np.intersect1d(
                np.where(
                    raw_acq.adc_input == input_number),
                np.where(
                    raw_acq.crate == crate_number),
                np.where(
                    raw_acq.slot == slot_number)
            )]
            single_inp.time_streams = raw_acq.timestream[np.intersect1d(
                np.where(
                    raw_acq.adc_input == input_number),
                np.where(
                    raw_acq.crate == crate_number),
                np.where(
                    raw_acq.slot == slot_number))]
            input_id = {}
            input_id["crate"] = crate_number 
            input_id["slot"] = slot_number
            input_id["input"] = input_number
            single_inp.input_id = input_id
    
        def get_rms_std(single_inp):
            '''
            
            Input: an array (passed from init())
            Outputs: n/a
            
            This function defines the following values: standard deviation of the acd data and the root-mean-square
            of the acd data. 
            
            This function is automatically called when the check_input() object gets initialized.
            
            '''
            istream = single_inp.time_streams
            adc_std = np.std(istream, axis=1)
            adc_rms =  np.sqrt(np.mean(np.square(istream), axis = 1))
            single_inp.adc_std = adc_std
            single_inp.adc_rms = adc_rms

        def get_fft_of_adc_counts(single_inp):
            '''
            
            Input: an array (passed from init())
            Outputs: n/a
            
            This function defines the following values: the fast Fourier Transform of the time stream data (quantized sine waves) which 
            which converts the data from position space to frequency space, the magnitude of the fast Fourier Transform, and the angles 
            associated with the fast Fourier Transform in units of radians.
            
            This function is automatically called when the check_input() object gets initialized.
            
            '''
            istream = single_inp.time_streams
            window = get_window('blackmanharris',2048)
            ffted_data = np.fft.fft(istream*window, axis=1)
            single_inp.fft = ffted_data[:,:ffted_data.shape[1] // 2]
            single_inp.mag_fft = np.abs(ffted_data)[:,:ffted_data.shape[1] // 2]
            single_inp.angle_fft = np.angle((ffted_data)[:,:1024])
            
        
        def get_single_input_rms(single_inp):
            '''
            
            Input: an array (passed from init())
            Outputs: n/a
            
            This function is automatically called when the check_input() object gets initialized.            
            
            '''
            istream = single_inp.time_streams
            single_inp.rms = np.sqrt(np.mean(np.square(istream), axis = 1))

        def get_fgpa_count_for_input(single_inp):
            '''
            
            Input: an array (passed from init())
            Outputs: n/a
            
            This function isolates the time at each fpga snapshot and saves it to its own list.
            
            This function is automatically called when the check_input() object gets initialized.           
            
            '''
            single_inp.time_fpga_count = single_inp.time_stamps["fpga_count"]
            
        def inspect_maser(single_inp):
            '''
            
            Input: an array (representing the input to check)
            Outputs: n/a
            
            This function first determines the index corresponding to 10 MHz, which is related to the 10 MHz clocks.
            Then, it defines the angles for each of those 10 MHz indeces accross all snapshots and calculates the 
            tau values for those angles. 
            
            '''
            tenMHz_index = int(np.round(10/(400/1024)))
            angles = single_inp.angle_fft[:,tenMHz_index]
            single_inp.angles = angles
            angles = np.unwrap(angles - angles[0])
            single_inp.tau = angles/2/np.pi/10e6 # angle/nu; tau in seconds
            
        def plot_single_input_diagnostics(single_inp):
            '''
            
            
            
            '''
            single_inp.get_rms_std()
            single_inp.get_fft_of_adc_counts()
            #########################################################################################################
            fig, axd = plt.subplot_mosaic([['rms'],
                                       ['fft']],
                                      figsize=(15, 10), constrained_layout=True)
            #########################################################################################################
            fig.suptitle(
                f"crate number.slot number.input_number = {single_inp.input_to_check[0]}.{single_inp.input_to_check[1]}.{single_inp.input_to_check[2]}")
            axd["rms"].set_title('root mean square of adc counts')
            axd["rms"].axhline([128], c = 'r')
            axd["rms"].axhline([0], c = 'r')
            axd["rms"].set_ylabel('rms')
            axd["rms"].set_xlabel('fpga count number')
            axd["rms"].scatter(single_inp.time_stamps['fpga_count'],single_inp.adc_rms)
            axd["fft"].set_xlabel('frequency (MHz)')
            axd["fft"].set_ylabel('fpga_count')
            axd["fft"].imshow(
                single_inp.mag_fft, 
                aspect='auto', 
                vmin = np.percentile(single_inp.mag_fft,5), 
                vmax = np.percentile(single_inp.mag_fft,95), 
                extent=[800, 400, single_inp.time_stamps['fpga_count'][-1], single_inp.time_stamps['fpga_count'][0]]
            )
            fig.show()
    
    class check_iceboard:
        """
            Check adc rms of all inputs of an iceboard of a given crate and slot from a singel raw_acq file
        """
        def __init__(iceboard, crate, slot): #, time_slice):
            '''
            
            
            
            '''
            #iceboard.time_slice = time_slice
            iceboard.crate = crate
            iceboard.slot = slot
            iceboard.full_acq_capture_diagnostic()
        
        def full_acq_capture_diagnostic(iceboard): 
            """
                reads all data from a single raw_acq file and computes rms and std and plots histgram of all the adc inputs
            """
            #if iceboard.time_slice: 
            #    timeslice = iceboard.time_slice
            ant_std = np.zeros(16)
            ant_rms = np.zeros(16)
            plt.figure(figsize=(15,8))
            plt.suptitle(f"total adc_rms of (crate,slot){iceboard.crate}{iceboard.slot} between {raw_acq.start_time} and {raw_acq.end_time}")
            #print("\n\n")
            #print("(crate,slot,input),rms,log2std")
            for i in range(16):
                inp0 = np.where(raw_acq.adc_input[:] == i)[0]
                ant0_data = raw_acq.timestream[:][inp0]
                ant0_data = ant0_data[:]
                #ant_rms[i] = np.sqrt(np.mean(ant0_data)**2)
                #ant_std[i] =  np.log2(np.std(ant0_data))
                #print(f"({check_crate},{check_slot},{i}),{ant_rms[i]:1.3f},{ant_std[i]:1.3f}")
                plt.subplot(4,4,i+1)
                hist, bin_edges = np.histogram(ant0_data, bins=256,  density=True)
                plt.plot(bin_edges[1:], hist)
                plt.title(f'input: {i}')
                plt.tight_layout()
            plt.show()

            
class analyse_maser: 
    def __init__(self, raw_acq_folder, maser_input, num_files = None):
        '''
        
        
        
        '''
        self.folder_path = raw_acq_folder
        self.maser_input = maser_input
        self.num_files = num_files
        self.read()
        print("DONE reading files and getting delays")
        #self.plot_delays()
        #self.get_allan_deviation()
    
    def read(self):
        '''
        
        
        
        '''
        files = glob.glob(self.folder_path + "*[!.lock]")
        files.sort()
        files = files[:self.num_files]
        print(*files, sep = "\n")
        taus = []
        delays = []
        angles = []
        num_files = len(files) ### this is zero
        #calling progressbar with it=0 so that when we initialize count = 0, we divide by 0 (in x)
        input_to_check = self.maser_input            
        for i in progressbar(range(num_files), "Computing Delay: ", 80):
            file_name = files[i]
            try:
                raw_acq(file_name)
            except OSError: 
                pass
            maser = raw_acq.check_input(input_to_check)
            maser.inspect_maser()
            taus.append(maser.time_fpga_count)
            delays.append(maser.tau)
            angles.append(maser.angles)
            
        self.fpgatime = np.concatenate(taus, axis = 0)
        self.angles = np.concatenate(angles, axis = 0)
        self.delays = np.concatenate(delays, axis = 0)
        self.angles = np.unwrap(self.angles)
        self.taus = self.angles/2/np.pi/10e6
       
      
    
    def plot_delays(self):
        '''
        
        
        
        '''
        weeks = self.fpgatime*2.56e-6/60/60/24/7
        timesaxis = weeks
        time_axis = "Weeks"
        if weeks.max() < 5:
            days = self.fpgatime*2.56e-6/60/60/24
            timesaxis = days
            time_axis = "Days"
            if days.max() < 5:
                hours = self.fpgatime*2.56e-6/60/60
                timesaxis = hours
                time_axis = "Hours"
                if hours.max() < 2: 
                    seconds = self.fpgatime*2.56e-6
                    timeaxis = seconds
                    time_axis = "seconds"
        plt.figure(figsize=(13,4))
        plt.scatter(timesaxis,(self.taus/1e-9), s= 0.1, c = 'k', marker = '.')
        plt.xlabel(time_axis)
        plt.ylabel(r" $\Delta(\tau)$ (ns)")
        #plt.savefig("figure/gpsvmaser.pdf",dpi = 300, format = "pdf", bbox_inches='tight')
        self.plt = plt
        
    def get_allan_deviation(self):
        '''
        
        
        
        '''
        taus_from_fpga_counts = self.fpgatime*2.56e-6
        (taus, adevs, errors, ns) = allan.oadev(self.delays, taus = taus_from_fpga_counts)
        self.adevs = adevs
        self.adev_taus = taus
        plt.figure(figsize=(6.5,5))
        plt.loglog(taus,adevs, c = 'k', lw = 1)
        plt.ylabel("Allan Deviation")
        plt.xlabel("Time (s)")
        plt.grid()
        #plt.savefig("figure/adev.pdf",dpi = 300, format = "pdf", bbox_inches='tight')
        self.plt = plt
    
        
def get_newest_file(folder_path):
    '''
    
    
    
    '''
    files = glob.glob(folder_path + "*[!.lock]")
    newest_file = max(files, key=os.path.getctime)
    return newest_file

def get_second_newest_file(folder_path):
    '''
    
    
    
    '''
    files = glob.glob(folder_path + "*[!.lock]")
    newest_file = max(files, key=os.path.getctime)
    files.remove(newest_file)
    newest_file = max(files, key=os.path.getctime)
    return newest_file