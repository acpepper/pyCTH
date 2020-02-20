# pyCTH


HOTCI
(HERCULES Output To CTH Input)
NOTES:
* HOTCI prints omega_rot but this value might not be correct since hercules cannot enforce both a constant angular momentum and a constant rotation rate
________________


Table of Contents
Table of Contents
1 Introduction
1.1 HERCULES
1.2 CTH
1.3 HOTCI
2 List of included files
2.1 HOTCI.py
2.2 HERCULES_structures.py
2.3 eos/EOS_class.h
2.4 eos/EOS_functions.cc
2.5 eos/setup.py
2.6 eos/vector_operations.h
2.7 eos/vector_operations.cc
3 Compiling and Running
3.1 Dependencies
3.2 Wrapping eos.so
3.3 Running
________________
1 Introduction
HOTCI is intended to be used to study giant impacts (impacts between planet-sized bodies) by allowing the user to initialize a rotating body in an Eulerian shock physics code. This is desirable because previous Eulerian simulations of giant impact have not accurately initialized rotating bodies. Furthermore most impact simulation that have high angular momentum have been conducted with one method, namely Smoothed Particle Hydrodynamics. The results of these must be validated using different methods.
1.1 HOTCI
        HOTCI is a small library written in Python and C++ that can be used to simulate rapidly rotating bodies in the shock physics code CTH. This is accomplished through a multistep process (see Figure 1).
1. A rapidly rotating body is generated using HERCULES (see section 1.2).
2. The output of HERCULES (a custom binary format) is read by HOTCI
3. The body is analyzed by HOTCI. During this step HOTCI might unresolve the body if the resolution in HERCULES is too high. HOTCI might also calculate the temperature of each HERCULES layer if the user desires.
4. The data that defines the body is converted into a string format.
5. A “blank” CTH input file is read by HOTCI (this file is not actually blank and must be a working CTH input file). HOTCI searches the file for the correct location to insert the simulations initial conditions. The initial conditions defined in the blank input file are then overwritten with the aforementioned string. HOTCI creates a new file such that the blank input file is undisturbed.
6. CTH reads the newly created input file and the simulation is carried out.
1.2 HERCULES
HERCULES (Highly Eccentric Rotating Concentric U [Potential] Layers Equilibrium Structure) is a program written by Simon Lock to solve for the equilibrium structure of a self-gravitating fluid. The algorithm used by HERCULES was originally found by Hubbard (2012, 2013) in order to study Jupiter. The algorithm has since been extended by Kong et al. (2013) and Hubbard et al. (2014) to accomodate bodies with large rotational distortion. HERCULES is an open-source manifestation of this algorithm, written in C++.
1.3 CTH
        CTH is a large shock-physics code that has been over-seen by many employees of Sandia National Laboratory. It is fundamentally an Eulerian method though at each time-step it solves the Lagrangian equations and remaps the solution to the Eulerian grid via a van Leer scheme that is accurate to second order (van Leer, 1977; McGlaun, 1982). CTH implements two major features that make it popular for simulating giant impacts. Firstly, it implements self-gravity, which is critical for studying any process in the large length regime. Secondly, it implements adaptive mesh refinement, wherein the Eulerian mesh is recursively subdivided to increase resolution locally, this saves computational resources when large regions of the simulation domain are occupied by the void of space.
  Figure 1. A schematic of how HOTCI works, note that only the topmost pictures contain real data the rest of the images have been rendered solely for illustrative purposes. (A) In this step HOTCI reads a HERCULES output file and converts it into a CTH input file. (B) CTH reads the input file and processes the body one layer at a time. Each layer is homogeneous in density, pressure, and temperature. (C) The layer is incorporated into the Eulerian mesh. In this step CTH gives each cell of the mesh a velocity, volume fraction for each material, and any necessary thermodynamic variables. (D) This panel is included to illustratculties one has when representing a spherical object in a rectangular grid, the resolution is exaggerated. (E) A cross section of an example body in CTH.
2 List of included files
2.1 HOTCI.py
A Python file containing HOTCI’s main function. There are several variables defined at the top of HOTCI.py that are intended to be edited by the user. These variables determine HOTCI’s reading and writing behavior. They are:
* HERCULES_OUT_DIR: A string containing the directory where HERCULES dumps its output files.
* HERCULES_OUT_FNAMES: A list of strings containing the names of the HERCULES output files that will be read. The user is able to include any number of file names however the length of the HERCULES_OUT_FNAMES list must be equal to that of … In order for HOTCI to run properly, the user is responsibility for ensuring that this condition is met.
* CTH_IN_DIR: A string containing the directory where HOTCI searches for and saves all CTH input files.
* CTH_BASE_FNAME: HOTCI requires a partial CTH input file to work from, this is a sting containing the name of such a file.
* CTH_IN_FNAME: A string containing the name of the CTH input file that HOTCI generates.
* MATERIAL_FNAMES: A list of strings containing the locations of 
2.2 HERCULES_structures.py
        A Python file containing classes for analyzing the binary output of HERCULES.
2.3 EOS_class.h (eos directory)
        A C++ file containing the EOS class definition. This class is used to calculate the temperature of the HERCULES layers.
2.4 EOS_functions.cc (eos directory)
        A C++ file containing function definitions for the EOS class.
2.5 setup.py (eos directory)
        A Python file that determines how the C++ files in the eos directory will be compiled into a Python library.
2.6 vector_operations.h (eos directory)
        A C++ header containing the VecDoub class definition.
2.7 eos/vector_operations.cc (eos directory)
        A C++ file containing the function definitions for the VecDoub class.
        
3 Compiling and Running
        All compiling instructions are for a Linux operating system. If this is not your operating system of choice, the instructions should be straightforward to translate since HOTCI’s compiling procedures are quite generic. 
3.1 Dependencies
HOTCI requires CTH and HERCULES to be installed and running. To work properly CTH should implement self-gravity, which has been included in the latest version for many years. HOTCI was written primarily in Python 2.7 and thus requires a Python 2.7 interpreter; it will not run on a Python 3.0 interpreter or later. The only element of HOTCI which must be compiled is the eos.so library, which is a wrapping of a C++ library that was written for HERCULES. Thus a C++ compiler will also be needed. There are many ways to create a python library by wrapping C++ source code. The method detailed here used the distutils and Cython libraries. These libraries are included in many of the most popular python distributions, including Anaconda and Sage, so they will likely be installed with the Python 2.7 interpreter. To check if the distutils and Cython libraries were included in your Python distribution run the following from the command line.


$ python
>>> from distutils.core import setup
>>> from Cython.Build import cythonize
	

If this does not produce an error than you are ready to start compiling HOTCI.
3.2 Wrapping eos.so
        From the HOTCI directory, enter the eos subdirectory and run setup.py in “build” mode.


HOTCI$ cd eos
eos$ python setup.py build
	

This should create a new file called eos.so in the build subdirectory entitled lib.[your OS]. Copy the newly created eos.so file into the HOTCI parent directory.


eos$ cp build/lib.linux-x86_64-2.7/eos.so ../
	

        If the eos.so file was not created but distutils and Cython were properly installed, then the issue probably occurred when trying to link Python.h. To fix this error open the setup.py file and modify the include_dirs list to contain the directory where your Python.h file is located. On my machine this is the /opt/local/include/ directory.
3.3 Running
        Once the necessary libraries have been downloaded and compiled, the HOTCI.py file must be modified to match your work environment. This is accomplished by opening HOTCI.py file. Lines 17-38 contain all the variables a user might want to modify. They appear as follows.


HERCULES_OUT_DIR = "../Output/"
HERCULES_OUT_FNAMES = ["M96L1.5_L1.48725_N200_Nm800_k12_f020_p10000_l1_0_1.5_fi\
nal", "M12omega2e-4_L1.983_N100_Nm400_k12_f020_p10000_l1_0_1.5_final"]

CTH_IN_DIR = "CTH_in/"
CTH_BASE_FNAME = "CTH_ANEOS_test_impact.in"
CTH_IN_FNAME = "test_M91_m12_L1.5.in"

MATERIAL_FNAMES = ['../EOS_files/HERCULES_EOS_forsterite_S3.20c_log.txt', '../EOS_files/HERCULES_EOS_Iron_T4kK_P135GPa_Rho7-15.txt']
# PD_FLAG key:
# 1: pressure and temperature
# 2: density and temperature
# 3: pressure and density
PDT_FLG = 2

# NOTE: These are in CGS units
CENTERS = [[0, 0, 0], [7.056e8, 7.056e8, 0]]
VELOCITIES = [[0, 0, 0], [-8.795e5, 0, 0]]

# CTH limits the number of vertices in its input files so when the HERCULES
# resolution is too fine the shape cannot be transferred in a 1-to-1 fashion.
# When this occurs, we unresolve the HERCULES structure following a cubic
# spline interpolation of the original points. The new number of points is
# defined by NUM_NEW_MU.
NUM_NEW_MU = 600

INDENT = " "
	

Each variable’s usage is detailed in section 2.1. It is particularly important that the user updates their file names and directories.
        
        After the variables have been updated HOTCI can be run by simply typing the following into the command line.


HOTCI$ python HOTCI.py
	________________


Reviewed by Lisa Illes
1. Who do you think the audience is? How do you know?
   1. The end user of the HOTCI, based on the order in which the aspects of the HOTCI are presented and the content of the description as a whole. 
2. Does the author successfully provide you with an understanding of a process’/product’s parts?
   1. To an extent. The goal of this brief description seems to be to instruct the user when running this program, there is not as much discussion of how the program actually works.
3. Do you understand how all the parts work together?
   1. Yes, I understand how all the parts work together in the context of running the program. 
4. Is the information organized so you can easily understand how information is grouped? Are there headings and are they descriptive? How might the author consider ordering/chunking information differently?
   1. Yes, I think that the information is organized in a logical manner that is easy for the reader to follow. 
5. Are there visuals? Are the visuals clearly explained?
   1. If you consider screenshots of code to be a visual, then yes. Otherwise, no. 
6. Does it appear that the student cited all borrowed information? Or, is the student at risk of being accused of plagiarizing?
   1. No. The description might benefit from a bibliography, since several papers are mentioned in the introduction. 
7. Is the document well designed, is there appropriate uses of headings, subheadings, paragraphs, white space, font, color, and lists?
   1. Yes. 


Feedback
1. What I liked about the description is:
   1. It’s concise and to the point. I understand what you are describing and in what particular situation it is useful.
2. What can be improved is:
   1. Include more in depth descriptions of the individual variables and provide additional context and definitions for the variables
3. Steps I’d take to improve this description include:
* Add descriptions of the variables
* Describe what programming environment this can be run in a bit more
* Consider adding an example with some simulated data
