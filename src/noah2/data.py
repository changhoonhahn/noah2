'''

module to interface with data


'''
import os
import numpy as np
from astropy.table import Table as aT


class Noah2(object): 
    ''' data object for NOAH2 project
    '''
    def __init__(self, version='0.1'): 
        self.version = version 

    def data(self, version='0.1', sample='all', columns='all', unique_zipcode=False): 
        ''' get data set with only the necessary columns and select subsamples

        params
        ------
        version : str 
            version of the data set 
        sample : str
            string specifying the subsample 
        columns : str
            string specifying the columns 
        unique_zipcode : bool 
            if True, only return unique zipcodes
            if False, leave it alone
        '''
        # read in full FEMA dataset 
        fema = self._read_data_full()

        # get sample from full dataset 
        if sample == 'all': 
            _sample = np.ones(len(fema)).astype(bool)
        elif sample == 'participants': 
            # only CRS participants 
            participants = self._participants(fema)
            fema = fema[participants]
        elif sample == 'non-participants': 
            # only non-CRS participants 
            participants = self._participants(fema)
            fema = fema[~participants]
        elif 'binary' in sample: 
            # communities with different combinations of binary CRS activities
            binary_code = int(sample.strip('binary')) 
            codes = self._crs_binary_codes(fema)
            fema = fema[codes == binary_code]
        else: 
            raise ValueError

        if unique_zipcode: 
            keep = self.keep_unique_zipcodes(fema['zipcode'])
            fema = fema[keep]

        # get data columns 
        if columns == 'all': 
            columns = self._columns()
        elif columns == 'props': 
            # only community properties 
            columns = self._columns()[1:8]
        elif columns == 'props+': 
            # only community properties + some CRS
            columns = self._columns()[1:8] + ['s_c350', 's_c420', 's_c450']
        elif columns == 'oprops': 
            # outcome and community properties 
            columns = self._columns()[:8]
        elif columns == 'oprops+': 
            # outcome and community properties + some CRS
            columns = self._columns()[:8] + ['s_c350', 's_c420', 's_c450']
        elif columns == 'crs_lowimpact': 
            # only low impact CRS activities
            columns = self._crs_lowimpact_activities()
        elif columns == 'crs_binary': 
            # only binary CRS activities
            columns = self._crs_binaryactivities()
        else: 
            raise ValueError("specify valid columns") 

        data = np.array([np.array(fema[col]) for col in columns]).T
        return data 
    
    def _read_data_full(self, version='0.1'): 
        ''' read in full dataset compiled in `nb/0_compile_data.ipynb`
        '''
        # read in full data set 
        fema = aT.read(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 
            'dat/noah2.v%s.csv' % version))
        return fema 

    def _participants(self, fema): 
        ''' identify the communities that participate in the CRS (i.e. actually
        scores in the CRS program)
        '''
        crs_acts = self._crs_activities()

        scores = np.array([np.array(fema[col]) for col in crs_acts]).T
        non_participant = np.all(scores == 0, axis=1)
        return ~non_participant 

    def _columns(self): 
        ''' column names of data set 
        '''
        return ['amountPaidOnTotalClaim_per_policy', 'mean_rainfall', 'avg_risk_score_all', 'median_household_income', 
                'population', 'renter_fraction', 'educated_fraction', 'white_fraction', 
                's_c310', 's_c320', 's_c330', 's_c340', 's_c350', 's_c360', 's_c370', 
                's_c410', 's_c420', 's_c430', 's_c440', 's_c450', 
                's_c510', 's_c520', 's_c530', 's_c540',
                's_c610', 's_c620', 's_c630'] 

    def _column_labels(self): 
        return ['Claim Per Policy', 'Mean Rainfall', 'Flood Risk', 'Median Income', 'Population', 'Renter Fraction', 'Educational Attainment', 'White Fraction', 'c310 Score', 'c320 Score', 'c330 Score', 'c340 Score', 'c350 Score', 'c360 Score', 'c370 Score', 'c410 Score', 'c420 Score', 'c430 Score', 'c440 Score', 'c450 Score', 'c510 Score', 'c520 Score', 'c530 Score', 'c540 Score', 'c610 Score', 'c620 Score', 'c630 Score'] 

    def _crs_activities(self): 
        return ['s_c310', 's_c320', 's_c330', 's_c340', 's_c350', 's_c360',
                's_c370', 's_c410', 's_c420', 's_c430', 's_c440', 's_c450',
                's_c510', 's_c520', 's_c530', 's_c540', 's_c610', 's_c620',
                's_c630']

    def _crs_lowimpact_activities(self): 
        return ['s_c310', 's_c320', 's_c330', 's_c340', 's_c370', 's_c410',
                's_c430', 's_c440', 's_c510']

    def _crs_binary_activities(self): 
        return  ['s_c360', 's_c520', 's_c530', 's_c540', 's_c610', 's_c620',
                 's_c630']

    def _crs_binary_codes(self, fema): 
        ''' get binary codes 
        '''
        binary_activities = self._crs_binary_activities()

        binary_data = np.array([np.array(fema[col]) for col in binary_activities]).T
        binary_data = (binary_data > 0).astype(int)

        binary_act_codes = np.zeros(len(fema))
        for i in range(binary_data.shape[1]):
            binary_act_codes += 2**i * binary_data[:,i]

        return binary_act_codes.astype(int) 

    def _uncode(self, num):
        ''' uncode binary code 
        '''
        binary_activities = self._crs_binary_activities()

        act = np.zeros(len(binary_activities)).astype(int)
        while num > 0:
            ind = int(np.floor(np.log(num)/np.log(2)))
            num = num - 2**ind
            act[ind] = 1
        return act

    def keep_unique_zipcodes(self, zipcodes): 
        ''' only keep unique zipcodes 
        '''
        zuniq, iuniq, nuniq = np.unique(zipcodes, return_index=True, return_counts=True)

        keep = np.zeros(len(zipcodes)).astype(bool)
        for z in zuniq:
            is_zip = (zipcodes == z)
            i_keep = np.random.choice(np.arange(np.sum(is_zip)), 1)
            
            keep[np.arange(len(zipcodes))[is_zip][i_keep]] = True
        return keep  

    def _binary_codes_enough_stat(self): 
        ''' get codes for the combination of binary CRS activities that have
        enough statistics
        '''
        # read data set 
        fema = self._read_data_full(version=self.version)
        participants = self._participants(fema)
        fema = fema[participants]

        binary_codes = self._crs_binary_codes(fema)

        act_enough_stat = []
        for code in np.unique(binary_codes):
            is_code = (binary_codes == code)
            ncomm = np.sum(is_code)
            nzip = len(np.unique(fema['zipcode'][is_code]))
            if nzip < 100: continue
            print(self._uncode(code))
            print('%i communities' % ncomm)
            print('%i unique zipcodes' % nzip)
            print()
            act_enough_stat.append(code)

        return np.array(act_enough_stat) 
