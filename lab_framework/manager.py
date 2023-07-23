''' manager.py

Class for managing the automated laboratory equiptment, including data collection and motor manipulation.

Author(s):
- Alec Roberson (aroberson@hmc.edu) 2023
'''

# python imports
import json
import time
import os
import datetime
import copy
from typing import Union, List, Tuple
import serial

# package imports
from tqdm import tqdm
import numpy as np
from uncertainties import unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat
import pandas as pd

# local imports
from .monitors import CCU # , Laser # soon!
from .motor_drivers import MOTOR_DRIVERS

# manager class

class Manager:
    ''' Class for managing the automated laboratory equiptment.

    Parameters
    ----------
    config : str, optional
        The name of the configuration file to load. Defaults to 'config.json'.
    motors : bool, optional (default=True)
        Initialize motors?
    ccu : bool, optional (default=True)
        Initialize CCU?
    laser : bool, optional (default=True)
        Initialize laser?
    debug : bool, optional (default=False)
        If True, do not initialize any equipment. Useful for debugging. All not initialized equiptment can still be initialized via init_motors, init_ccu, and init_laser.
    verbose : bool, optional (default=True)
        If True, print all log messages to output.
    '''
    def __init__(self, config:str='config.json', motors:bool=False, ccu:bool=False, laser:bool=False, debug:bool=False, verbose:bool=True):
        # get the time of initialization for file naming
        self._init_time = time.time()
        self._init_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # initialize the log file
        removed_log = False
        if os.path.isfile('./mlog.txt'):
            removed_log = True
            os.remove('./mlog.txt')

        # initialize the log file
        self._log_file = open('./mlog.txt', 'w+')
        self._verb = verbose
        self.log(f'Manager started at {self._init_time_str}.', self._verb)
        if removed_log: self.log('Removed old log file.', self._verb)
        self.log(f'Configuration file: "{config}".', self._verb)

        # check config path
        if not os.path.isfile(config):
            self.log(f'Configuration file "{config}" does not exist.', self._verb)
            self.shutdown()
            raise ValueError(f'Configuration file "{config}" does not exist.')
        
        # load the configuration file
        self.log('Loading configuration file.', self._verb)
        with open(config, 'r') as f:
            self._config = json.load(f)

        # initialize ccu, motor, and laser
        self._ccu = None
        self._motors = []
        self._laser = None
        if not debug:
            if motors: self.init_motors()
            if ccu: self.init_ccu()
            if laser: self.init_laser()

        # initialize output file variables
        self._output_data = pd.DataFrame(columns=self.df_columns)

    # +++ class variabes +++

    MAIN_CHANNEL = 'C4' # the main coincidence counting channel key for any preset basis

    # +++ initialization methods +++

    def init_ccu(self) -> None:
        ''' Initialize the CCU which starts live plotting. '''
        self.log('Beginning CCU initialization...', self._verb)

        if self._ccu is not None:
            self.log('CCU already initialized. Aborting initialization.', self._verb)
            return

        # initialize the ccu
        try:
            self._ccu = CCU(
                port=self._config['ccu']['port'],
                baud=self._config['ccu']['baudrate'],
                plot_xlim=self._config['ccu'].get('plot_xlim', 60),
                plot_smoothing=self._config['ccu'].get('plot_smoothing', 0.5),
                channel_keys=self._config['ccu']['channel_keys'],
                ignore=self._config['ccu'].get('ignore', []))
        except Exception as e:
            self.log(f'CCU ran into exception during initialization: {e}', self._verb)
            self.shutdown()
    
    def init_motors(self) -> None:
        ''' Initialize and connect to all motors. '''
        self.log('Beginning motor initialization...', self._verb)

        # intialize the active ports
        self._active_ports = {}

        if len(self._motors):
            self.log('Motors already initialized, aborting initialization.', self._verb)
            return

        self._motors = list(self._config['motors'].keys())
        self.log(f'Initializing motors from config: {", ".join(self._motors)}.', self._verb)

        # loop to initialize all motors
        for motor_name in self._motors:
            # check name
            if motor_name in self.__dict__ or not motor_name.isidentifier():
                self.log(f'Motor initializer "{motor_name}" is invalid.', self._verb)
                self.log('Aborting motor initialization.', self._verb)
                self.shutdown_motors()
                return
            # get the motor arguments
            motor_dict = copy.deepcopy(self._config['motors'][motor_name])
            typ = motor_dict.pop('type')
            # conncet to com ports for elliptec motors
            if typ == 'Elliptec':
                port = motor_dict.pop('port')
                if port in self._active_ports:
                    com_port = self._active_ports[port]
                else:
                    self.log(f'Connecting to com port "{port}".', self._verb)
                    try:
                        com_port = serial.Serial(port, timeout=5) # time out for all read
                    except Exception as e:
                        self.log(f'Failed to connect to com port "{port}": {e}', self._verb)
                        self.shutdown_motors()
                        return
                    self._active_ports[port] = com_port
                motor_dict['com_port'] = com_port
            # initialize motor
            try:
                self.__dict__[motor_name] = MOTOR_DRIVERS[typ](name=motor_name, **motor_dict)
            except Exception as e:
                self.log(f'Failed to initialize motor "{motor_name}": {e}', self._verb)
                self.shutdown_motors()
                return
        self.log('Motor initialization complete.')

    def init_laser(self) -> None:
        ''' Initializes laser monitor. '''
        self.log('Laser module not operational.', self._verb)

    # +++ class methods +++

    @staticmethod
    def reformat_ufloat_to_float(df_:pd.DataFrame) -> pd.DataFrame:
        ''' Reformat a dataframe with ufloats to only contain floats.

        Each column "X" that has a uncertainties.core.Variable dtype will be broken into the columns "X" and "X_SEM".

        Parameters
        ----------
        df_ : pd.DataFrame
            The dataframe to reformat.
            
        Returns
        -------
        pd.DataFrame
            The reformatted dataframe.
        '''
        # create a copy to work with
        df = df_.copy()
        # loop through coluns
        for c in df.columns:
            # check if the column is a ufloat
            if df[c].apply(lambda x : isinstance(x, ucore.Variable)).all():
                # break into sems and values
                df[c+'_SEM'] = unp.std_devs(df[c])
                df[c] = unp.nominal_values(df[c])
                # convert dtypes
                df[c] = df[c].astype(float)
                df[c+'_SEM'] = df[c+'_SEM'].astype(float)
        return df

    @staticmethod
    def reformat_float_to_ufloat(df_:pd.DataFrame) -> pd.DataFrame:
        ''' Reformat a dataframe with floats to contain ufloats where applicable.

        Each column "X" that has a corresponding column "X_SEM" will be collapsed to the single column "X" containing ufloats.

        Parameters
        ----------
        df_ : pd.DataFrame
            The dataframe to reformat.
            
        Returns
        -------
        pd.DataFrame
            The reformatted dataframe.
        '''
        # create a copy of the dataframe to work with
        df = df_.copy()
        # loop through columns to see which should be reformatted
        to_reformat = [c for c in df.columns if c+'_SEM' in df.columns]
        # loop through columns to reformat
        for c in to_reformat:
            # recast to object type
            df[c] = df[c].astype(object)
            # create the ufloats
            for i in range(len(df)):
                df.at[i, c] = ufloat(df[c][i], df[c+'_SEM'][i])
        # drop the sem columns
        df.drop(columns=[c+'_SEM' for c in to_reformat], inplace=True)
        # return the new dataframe
        return df

    @staticmethod
    def load_data(file_path:str) -> pd.DataFrame:
        ''' Load data saved by this class directly into a pandas dataframe.

        Parameters
        ----------
        file_path : str
            The path to the csv data file to load.
        
        Returns
        -------
        pd.DataFrame
            The data loaded from the file.
        '''
        # start by just loading the data
        df = pd.read_csv(file_path)
        # return a reformatted version with ufloats
        return Manager.reformat_float_to_ufloat(df)

    # +++ properties +++
    
    @property
    def motor_list(self) -> 'list[str]':
        ''' List of the string names of all motors. '''
        return self._motors
    
    @property
    def time(self) -> str:
        ''' String time since initalizing the manager, like "hh:mm:ss". '''
        return str(datetime.timedelta(seconds=int(time.time()-self._init_time)))
    
    @property
    def now(self) -> str:
        ''' The current date and time like "YYYY.MM.DD hh:mm:ss".'''
        return datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")

    @property
    def df_columns(self) -> List[str]:
        ''' The columns of output data frames. '''
        out = ['start', 'stop', 'num_samp', 'samp_period']
        out += self._motors
        if self._ccu: out += self._ccu._channel_keys
        if self._laser: out += self._laser._channel_keys
        out += ['note']
        return out

    @property
    def csv_columns(self) -> List[str]:
        ''' The columns of output csv files. '''
        out = ['start', 'stop', 'num_samp', 'samp_period']
        out += self._motors
        if self._ccu: out += self._ccu._channel_keys
        if self._ccu: out += [f'{k}_SEM' for k in self._ccu._channel_keys]
        if self._laser: out += self._laser._channel_keys
        if self._laser: out += [f'{k}_SEM' for k in self._laser._channel_keys]
        out += ['note']
        return out

    # +++ file management +++

    def reset_output(self) -> None:
        ''' Clear (erase) the data in the current output data frame. '''
        self.log('Clearing output data.', self._verb)

        # just create a brand new output data frame
        self._output_data = pd.DataFrame(columns=self.df_columns)

    def output_data(self, output_file:str=None, clear_data:bool=True) -> pd.DataFrame:
        ''' Saves the output data to a specified file csv file.

        Parameters
        ----------
        output_file : str, optional
            The name of the csv file to save the data to. Default is None, in which case no data is saved.
        clear_data : bool, optional
            If True, the output data will be cleared after saving. Default is True.
        
        Returns
        -------
        pd.DataFrame
            The data being output.
        '''

        # create the csv dataframe and save to the output file
        if output_file is not None:
            self.log(f'Saving data to "{output_file}".', self._verb)
            csv_df = Manager.reformat_ufloat_to_float(self._output_data)
            csv_df = csv_df[self.csv_columns]
            csv_df.to_csv(output_file, index=False)
        
        # create a copy of the data to return
        df = self._output_data.copy()

        # clear the output data
        if clear_data:
            self.reset_output()
        
        # return the dataframe
        return 

    # +++ methods +++

    def take_data(self, num_samp:int, samp_period:float, *keys:str, note:str="") -> Union[np.ndarray, ucore.Variable, str, float, int]:
        ''' Take detector data

        The data is written to the csv output table.

        Parameters
        ----------
        num_samp : int
            Number of samples to take.
        samp_period : float
            Collection time for each sample, in seconds. Note that this will be rounded to the nearest 0.1 seconds (minimum 0.1 seconds).
        *keys : str
            Any channel keys (probably CCU keys, but any are allowed) to return data for. If no keys are given, all rates will be returned.
        note : str, optional (default "")
            A note can be provided to be written to this row in the output table which can help you remember why you took this data.
        
        Returns
        -------
        numpy.ndarray
            Array of values for the data taken during the last collection period.

        or
        
        ucore.Variable, str, float, int
            If only one channel is specified, a single value is returned (e.g. ufloat for CCU data; str for time data; float for motor position).
        '''
        # log the data taking
        row = len(self._output_data)
        self.log(f'Taking data for row {row}; sampling {num_samp} x {samp_period} s.', self._verb)
        
        # check for note to log
        if note != "":
            self.log(f'\tNote: "{note}"', self._verb)
        else:
            note = ""

        # add basic info data to the output
        self._output_data.at[row, 'num_samp'] = num_samp
        self._output_data.at[row, 'samp_period'] = samp_period
        self._output_data.at[row, 'note'] = note
        
        # record all motor positions
        for m in self._motors:
            self._output_data.at[row, m] = self.__dict__[m].pos
        
        # record start time
        self._output_data.at[row, 'start'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # send data requests
        if self._ccu:
            self.log('Requesting CCU data...', self._verb)
            self._ccu.request_data(num_samp, samp_period)
        if self._laser:
            self.log('Requesting Laser data...', self._verb)
            self._laser.request_data(num_samp, samp_period)

        # collect data
        if self._ccu:
            self.log('Receiving CCU data...', self._verb)
            ccu_data = unp.uarray(*self._ccu.acquire_data())
        if self._laser:
            self.log('Receiving Laser data...', self._verb)
            laser_data = unp.uarray(*self._laser.acquire_data())
        
        # record stop time
        self._output_data.at[row, 'stop'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # add ccu data to the output
        if self._ccu:
            for k, v in zip(self._ccu.channels, ccu_data):
                self._output_data.at[row, k] = v
        if self._laser:
            for k, v in zip(self._laser.channels, laser_data):
                self._output_data.at[row, k] = v

        # put together the output
        if len(keys) == 0:
            return None
        elif len(keys) == 1:
            return self._output_data[keys[0]][-1]
        else:
            return np.array(self._output_data[k][-1] for k in keys)

    def log(self, note:str, print_note:bool=True) -> None:
        ''' Log a note to the manager's log file.

        All notes in the file are timestamped from the manager's initialization.

        Parameters
        ----------
        note : str
            The note to log.
        print_note : bool, optional (default True)
            If True, the note will also be printed to the console.
        '''
        line = self.time + f'\t{note}'
        if not self._log_file.closed:
            self._log_file.write(line + '\n')
            if print_note: print(line)
        else:
            print(f'WARNING: Log file is closed. Cannot log "{line}".')

    def configure_motors(self, **kwargs) -> List[float]:
        ''' Configure the position of multiple motors at a time

        Parameters
        ----------
        **kwargs : <NAME OF MOTOR> = <GOTO POSITION DEGREES>
            Assign each motor name that you wish to move the absolute angle to which you want it to move, in degrees.
        
        Returns
        -------
        list[float]
            The actual positions of the motors after the move, in the order provided.
        '''
        # loop to ask each motor to move
        for motor_name, position in kwargs.items():
            if not motor_name in self._motors:
                self.log(f'Asked to move unknown motor "{motor_name}" to {position} degrees. Skipping.', self._verb)
                continue
            self.log(f'Moving motor "{motor_name}" to {position} degrees.', self._verb)
            self.__dict__[motor_name].goto(position, block=False)
        # loop to wait for each motor to finish
        out = []
        for motor_name in kwargs:
            if not motor_name in self._motors:
                continue
            p = self.__dict__[motor_name].pos
            self.log(f'Motor "{motor_name}" moved to {p} degrees.', self._verb)
            out.append(p)
        return out

    def configure_motor(self, motor:str, position:float) -> float:
        ''' Configure a single motor using a string key.
        
        Parameters
        ----------
        motor : str
            The name of the motor, provided as a string.
        position : float
            The target position for the motor in degrees.
        
        Returns
        -------
        float
            The actual position of the motor after the move.
        '''
        if not motor in self._motors:
            self.log(f'Asked to move unknown motor "{motor}" to {position} degrees. Skipping.', self._verb)
            return 0
        self.log(f'Moving motor "{motor}" to {position} deg.', self._verb)
        self.__dict__[motor].goto(position)
        out = self.__dict__[motor].pos
        self.log(f'Motor "{motor}" moved to {out} deg.', self._verb)
        return out

    def meas_basis(self, basis:str) -> None:
        ''' Set the measurement basis for Alice and Bob's half and quarter wave plates. 
        
        Parameters
        ----------
        basis : str
            The measurement basis to set, should have length two. All options are listed in the config.
        '''
        self.log(f'Loading measurement basis "{basis}" from config file.')
        # setup the basis
        A, B = basis
        self.configure_motors(
            A_HWP=self._config['basis_presets']['A_HWP'][A],
            A_QWP=self._config['basis_presets']['A_QWP'][A],
            B_HWP=self._config['basis_presets']['B_HWP'][B],
            B_QWP=self._config['basis_presets']['B_QWP'][B])

    def make_state(self, state:str) -> None:
        ''' Create a state from presets in the config file
        
        Parameters
        ----------
        state : str
            The state to create, one of the presets from the config file.
        '''
        # setup the state
        self.log(f'Loading state preset "{state}" from config file -> {self._config["state_presets"][state]}.')
        self.configure_motors(**self._config['state_presets'][state])

    def sweep(self, component:str, pos_min:float, pos_max:float, num_steps:int, num_samp:int, samp_period:float) -> Tuple[np.ndarray, np.ndarray]:
        ''' Sweeps a component of the setup while collecting data
        
        Parameters
        ----------
        component : str
            The name of the component to sweep. Must be a motor name.
        pos_min : float
            The minimum position to sweep to, in degrees.
        pos_max : float
            The maximum position to sweep to, in degrees.
        num_steps : int
            The number of steps to perform over the specified range.
        num_samp : int
            Number of samples to take at each step.
        samp_period : float
            The period of each sample, in seconds (rounds down to nearest 0.1s, min 0.1s).
        
        Returns
        -------
        np.ndarray
            Coincidence count rates over the sweep.
        np.ndarray
            Uncertainties in said coincidence count rates.
        '''
        self.log(f'Sweeping {component} from {pos_min} to {pos_max} degrees in {num_steps} steps.')
        # open output
        out = []
        # loop to perform the sweep
        positions = np.linspace(pos_min, pos_max, num_steps)
        for pos in tqdm(positions):
            self.configure_motors(**{component:pos})
            x = self.take_data(num_samp, samp_period, Manager.MAIN_CHANNEL)
            out.append(x)
        return positions, np.array(out)

    # +++ shutdown methods +++
    def shutdown_motors(self) -> None:
        ''' Shuts down all the motors, returning them to their home positions and closing connections with them. '''
        self.log('Shutting down motors.')

        if len(self._motors) == 0:
            self.log('NOTE: No motors are active.',self._verb)
        else:
            # loop to delete motors
            for motor_name in self._motors:
                # check that it exists
                if motor_name not in self.__dict__:
                    self.log(f'WARNING: Motor "{motor_name}" not found in manager variables. Skipping.', self._verb)
                    continue
                self.log(f'Shutting down {motor_name}.',self._verb)
                # get the motor object
                motor = self.__dict__[motor_name]
                # return to home position
                motor.hardware_home()
                self.log(f'{motor.name} returned to true position {motor.true_position} degrees.',self._verb)
                self.log(f'Deleting {motor.name} object.',self._verb)
                del self.__dict__[motor_name]

        # com ports
        if len(self._active_ports) == 0:
            self.log('NOTE: No com ports are active.', self._verb)
        else:
            # loop to shutdown ports
            for port in self._active_ports.values():
                self.log(f'Closing COM port: {port}.', self._verb)
                port.close()

    def shutdown_ccu(self) -> None:
        ''' Shutdown the CCU subprocess. '''
        self.log('Shutting down CCU.')
        if self._ccu:
            self._ccu.shutdown()
            self._ccu = None
            self.log('CCU shutdown complete.')
        else:
            self.log('NOTE: No CCU active.', self._verb)

    def shutdown_laser(self) -> None:
        ''' Shutdown the laser subprocess. '''
        self.log('Shutting down laser.')
        if self._laser:
            self._laser.shutdown()
            self._laser = None
            self.log('Laser shutdown complete.')
        else:
            self.log('NOTE: No laser active.', self._verb)

    def shutdown(self) -> None:
        ''' Shutsdown all the motors and terminates CCU processes, closing all com ports.
        '''
        self.log('Beginning shutdown procedure.', self._verb)
        
        self.shutdown_motors()
        self.shutdown_ccu()
        self.shutdown_laser()

        # output data
        if len(self._output_data['start']) != 0:
            self.log(f'WARNING: Shutting down with {len(self._output_data["start"])} rows of (potentially) unsaved data.')

        # log file
        self.log('Closing log file.')
        self._log_file.close()
