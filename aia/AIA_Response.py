
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup
import warnings
from astropy import time


class AIAEffectiveArea:

    def __init__(self, url='https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response/', filename=None):
        '''
        :param url: the online location of the response table
        :param filename: string optional location of a local response table to read, overrides url

        Usage
        aia_effective_area = AIAEffectiveArea()
        effective_area_ratio = aia_effective_area.effective_area_ratio(171, '2010-10-24 15:00:00')
        '''

        # Local input possible else fetch response table file from GSFC mirror of SSW
        if filename is not None:
            response_table = filename
        else:
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            all_versions = [node.get('href') for node in soup.find_all('a') if
                            node.get('href').endswith('_response_table.txt')]
            latest_table_url = url + sorted([table_files for table_files in all_versions if
                                             table_files.startswith('aia_V')])[-1]
            tbl = requests.get(latest_table_url).content
            response_table = io.StringIO(tbl.decode('utf-8'))

        # Read in response table
        self.response_table = pd.read_csv(response_table, sep='\s+', parse_dates=[1], infer_datetime_format=True, index_col=1)

    def effective_area(self, wavelength, time):
        '''
        :param wavelength: float wavelength of the aia target image
        :param time: string in a format to be read by pandas.to_datetime; the time of the aia target image
        :return: the effective area of the AIA detector interpolated to the target_time
        '''

        eff_area_series = self._parse_series(wavelength.value)

        if (pd.to_datetime(time) - eff_area_series.index[0]) < pd.Timedelta(0):
            warnings.warn('The target time requested is before the beginning of AIA', UserWarning)

        return time_interpolate(eff_area_series, time)

    def effective_area_ratio(self, wavelength, time):
        '''
        :param wavelength: float wavelength of the aia target image
        :param time: string in a format to be read by pandas.to_datetime; the time of the aia target image
        :return: the ratio of the current effective area to the  pre-launch effective area
        '''

        eff_area_series = self._parse_series(wavelength.value)

        launch_value = eff_area_series[eff_area_series.index.min()]

        if (pd.to_datetime(time) - eff_area_series.index[0]) < pd.Timedelta(0):
            warnings.warn('The target time requested is before the beginning of AIA', UserWarning)

        return time_interpolate(eff_area_series, time) / launch_value

    def _parse_series(self, wavelength):

        # Parse the input response table and return a pd.Series

        eff_area_series = self.response_table[self.response_table.WAVELNTH == wavelength].EFF_AREA

        current_estimate = eff_area_series[eff_area_series.index.max()]

        # Add in a distant future date to keep the interpolation flat
        eff_area_series = eff_area_series.reindex(pd.to_datetime(list(eff_area_series.index.values) +
                                                                 [pd.to_datetime('2040-05-01 00:00:00.000')]))
        eff_area_series[-1] = current_estimate

        return eff_area_series


def time_interpolate(ts, target_time):
    """

    :param ts: pandas.Series object with a time index
    :param target_time: string in a format to be read by pandas.to_datetime; the time to be interpolated to
    :return: value of the same type as in ts interpolated at the target_time

    Usage

    new_value = time_interpolate(time_series, '2010-10-24 15:00:00')

    """
    ts1 = ts.sort_index()
    b = (ts1.index > target_time).argmax()  # index of first entry after target
    s = ts1.iloc[b-1:b+1]

    # Insert empty value at target time.
    s = s.reindex(pd.to_datetime(list(s.index.values) + [pd.to_datetime(target_time)]))

    # Linear interpolation is the most logical
    return s.interpolate('time').loc[target_time]
