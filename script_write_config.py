################### scan on angspeed - delta_az plane

config_file_name = 'angspeed-delta_az.cfg'

# scan on angspeed from 0.1 to 3. with step 0.1
angspeed = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
delta_az = [20., 23., 26., 29., 32., 35., 38., 41., 44., 47., 50.]

# don't scan on other ss parameters
time_on_const_elevation = 120 # min
angspeed_psi = 0. # stay on constant psi
maxpsi = 0.
hwp_div = 8 # positions between 0 and 90 degrees
dead_time = 5

sampling_period = 0.05
nep = 4.7e-17
nep_normalization = '1year'
fknee = 1 # Hz

nside = 256
nrealizations = 10

################### write config file

import ConfigParser

config = ConfigParser.RawConfigParser()

config.add_section('Scanning_Strategy')
config.set('Scanning_Strategy', 'angspeed', angspeed)
config.set('Scanning_Strategy', 'delta_az', delta_az)
config.set('Scanning_Strategy', 'time_on_const_elevation', time_on_const_elevation)
config.set('Scanning_Strategy', 'angspeed_psi', angspeed_psi)
config.set('Scanning_Strategy', 'maxpsi', maxpsi)
config.set('Scanning_Strategy', 'hwp_div', hwp_div)
config.set('Scanning_Strategy', 'dead_time', dead_time)

config.add_section('Observation')
config.set('Observation', 'sampling_period', sampling_period)
config.set('Observation', 'nep', nep)
config.set('Observation', 'nep_normalization', nep_normalization)
config.set('Observation', 'fknee', fknee)

config.add_section('Analysis')
config.set('Analysis', 'nside', nside)
config.set('Analysis', 'nrealizations', nrealizations)

with open(config_file_name, 'wb') as configfile:
    config.write(configfile)
