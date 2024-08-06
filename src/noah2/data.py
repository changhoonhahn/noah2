'''

module to interface with data


'''
import os
import numpy as np
from astropy.table import Table as aT


class Noah2(object): 
    ''' data object for NOAH2 project
    '''
    def __init__(self): 
        self.data = self._read_data()

    def _read_data(self): 
        ''' read data set with only the necessary columns 
        '''
        fdata = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'dat/noah2.data.npy')

        if not os.path.isfile(fdata):  
            # read full FEMA dataset 
            fema = self._read_data_full(participants=True) 
        
            # get data columns 
            columns = self._columns()

            data = np.array([np.array(fema[col]) for col in columns]).T

            np.save(fdata, data) 
        else: 
            data = np.load(fdata)

        return data 
    
    def _read_data_full(self, participants=True): 
        ''' read in full dataset compiled in `nb/0_compile_data.ipynb`

        params
        ------
        participants : bool
            If True, only select CRS participants (scored more than 0 in any
            CRS activity). If False, select all.  

        '''
        # read in full data set 
        if participants: 
            fema = aT.read(
                    os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 
                                     'dat/noah2.participants.v0.0.csv'))
        else: 
            fema = aT.read(
                    os.path.join(
                                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 
                                     'dat/noah2.v0.0.csv'))
        return fema 

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
        return ['Claim Per Policy', 'Mean Rainfall', 'Flood Risk', 'Median Income', 'Population',
                'Renter Fraction', 'Educational Attainment', 'White Fraction', 
                'c310 Score', 'c320 Score', 'c330 Score', 'c340 Score', 'c350 Score', 'c360 Score', 'c370 Score', 
                'c410 Score', 'c420 Score', 'c430 Score', 'c440 Score', 'c450 Score', 
                'c510 Score', 'c520 Score', 'c530 Score', 'c540 Score',
                'c610 Score', 'c620 Score', 'c630 Score'] 
