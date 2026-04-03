# As the source, scatter plane and absorption plane are of different dimensions, each block of the arrays below is 1mm and multiplied with thier respective SiPM dimensions
# Scattering plane SiPMs are 2x2 mm
# Absorption plane SiPMs are 6x6mm

size_sourcePlane = 1
size_scatterPlane = 2
size_absorbPlane = 6

# Number of sources to be simulated
noOfSources = 1

# Number of events per source
noOfEvents = 200

file_name = "events_" + str(noOfSources) + "_" + str(noOfEvents)
ext = ".csv"

# ...........Constant values required to calculate Electron energy
# ..........(DO NOT CHANGE THESE VALUES)................

initialEnergy_eV = 1  # 1MeV
initialEnergy_J = 1.602176634 * (10**(-13))
h = 6.626 * (10**(-34))
eMass_eV = 0.5109989461  # 0.511 MeV/c^2
eMass_J = 9.109 * (10**(-31))
c = 3.0 * (10**8)
