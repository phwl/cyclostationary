# derived from https://gist.github.com/boylea/1a0b5442171f9afbf372

import numpy as np
import pyqtgraph as pg
import pyaudio
import pyqtgraph.ptime as ptime
from PyQt5 import QtCore, QtGui
import scf_fam

FS = 44100 #Hz
CHUNKSZ = 2048 #samples
dosave = False

class MicrophoneRecorder():
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.data = None
        self.stream = self.p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=FS,
                            input=True,
                            frames_per_buffer=CHUNKSZ)

    def read(self):
        # data = self.stream.read(CHUNKSZ)
        data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
        y = np.frombuffer(data, 'int16')
        self.data = y
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self):
        super(SpectrogramWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((1000, int(CHUNKSZ/2+1)))
        self.fps = 0
        self.updatetime = 0

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        # prepare window for later use
        self.win = np.hanning(CHUNKSZ)
        self.show()

    def update(self, chunk):
        # test code
        fchunk = np.array(chunk, dtype='float')
        Np = 256
        L = Np // 4
        self.img_array = np.absolute(scf_fam.fam(chunk, Np, L))

        self.img.setImage(self.img_array, autoLevels=False)

        # calc fps
        now = ptime.time()
        fps2 = 1.0 / (now-self.updatetime)
        self.updatetime = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        print("%0.1f fps" % self.fps)
        if dosave:
            smat = (Np, L, chunk, self.img_array)
            np.save('audiosample', smat)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='live scf plot')
    parser.add_argument("-s", action="store_true", help="save data file")
    args = parser.parse_args()
    if args.s:
        dosave = True

    app = QtGui.QApplication([])
    w = SpectrogramWidget()
    w.read_collected.connect(w.update)

    mic = MicrophoneRecorder(w.read_collected)

    # time (seconds) between reads
    interval = FS/CHUNKSZ
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(1000/interval) #QTimer takes ms

    app.exec_()
    mic.close()

