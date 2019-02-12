import AIA_Response
import warnings
import numpy as np
import pandas as pd

test_target_time = '2011-11-11 11:11:11'
test_wavelength = 304


def test_aia_eff_area(test_target_time, test_wavelength):

    output = AIA_Response.aia_eff_area(test_wavelength, test_target_time)
    assert output[0] < 1.

    try:
        warnings.simplefilter("ignore")
        output = AIA_Response.aia_eff_area(test_wavelength, '2000-01-01')
    except Warning:
        assert True
    assert np.isnan(output)


def test_time_interpolate(test_target_time):

    ts = pd.Series(np.arange(0.1, 0.6, 0.1),
                   index=pd.to_datetime(['2010-01-01', '2011-02-01', '2012-03-01', '2013-04-01', '2014-05-01']))
    output = AIA_Response.time_interpolate(ts, test_target_time)
    assert output[0] == 0.27194571054239525

