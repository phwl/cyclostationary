# cyclostationary 

Implementation of cyclostationary analysis:

requirements.txt - dependencies\
audiosample.npy  - sample array to verify computation\
cyclostationary.py - estimates Spectral Correlation Density (SCD) using FFT accumulation (FAM) method\
livescd.py       - reads from microphone and plots scd using pyqtgraph
intro.ipynb      - jupyter notebook showing SCD for different modulation types

To verify: 
~~~
python cyclostationary.py # should print "audiotest: PASS (error=0.0)" and show
~~~

![Screen shot](images/ScreenShot_scf_fam.png)

To acquire audio from microphone and plot scf:
~~~
python livescd.py
~~~

![Screen shot](images/ScreenShot_livescf.png)
