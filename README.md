# cyclostationary 

Implementation of cyclostationary analysis:

requirements.txt - dependencies\
audiosample.npy  - sample array to verify computation\
scf_fam.py       - estimates Spectral Correlation Function (SCF) using FFT accumulation (FAM) method\
livescf.py       - reads from microphone and plots scf using pyqtgraph

To verify: 
~~~
python scf_fam.py # should print "audiotest: PASS (error=0.0)" and show
~~~

![Screen shot](images/ScreenShot_scf_fam.png)

To acquire audio from microphone and plot scf:
~~~
python livescf.py
~~~

![Screen shot](images/ScreenShot_livescf.png)
