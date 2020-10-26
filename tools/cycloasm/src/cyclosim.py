import numpy as np
import argparse as argparse

NREGS=256        # number of registers

# cyclostationary processor class
class cyclo:
    def __init__(self, fname):
        self.pc = 0
        self.x = np.zeros((2, NREGS), dtype=np.int16)
        self.mem = [int(h, 16) for h in open(fname).readlines()]

    def dump(self):
        print('pc={}'.format(self.pc))
        print('reg={}'.format(self.x))
        print('mem={}'.format(self.mem))

    def step():
        ir = self.mem[self.pc]	# fetch instruction
		op = ir & 0xff000000)

def main(fname):
    c = cyclo(fname)
    c.dump()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()
    main(args.fname)

