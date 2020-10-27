#!/usr/bin/python3

import argparse as argparse

class Carith:
    # truncation of 32b integer to 16b with sign extension
    def trunc(a):
        r = a
        if (a & 0x8000):
            r = r | 0xffff0000
        else:
            r = r & 0x0000ffff
        return r
    
    def illegal(rs1, rs2, rs3): 
        print("Illegal instruction")
    
    def add(a, b, c): 
        return(Carith.trunc(a[0] + b[0]), Carith.trunc(a[1] + b[1]))

    def sub(a, b, c): 
        return(Carith.trunc(a[0] - b[0]), Carith.trunc(a[1] - b[1]))

    def mul(a, b, c): 
        return(Carith.trunc(a[0] * b[0] - a[1] * b[1]), 
               Carith.trunc(a[0] * b[1] + a[1] * b[0]))

    def max(a, b, c): 
        if (a[0]*a[0] + a[1]*a[1] > b[0]*b[0] + b[1]*b[1]):
            return(a)
        else:
            return(b)
    
    # b + a*c
    def mulsub(a, b, c): 
        m = Carith.mul(a, c, b)
        return(Carith.trunc(b[0] - m[0]), Carith.trunc(b[1] - m[1]))
    
    # b - a*c
    def muladd(a, b, c): 
        m = Carith.mul(a, c, b)
        return(Carith.trunc(b[0] + m[0]), Carith.trunc(b[1] + m[1]))
    
    def exec(i, a, b, c):
        OP_F= [Carith.illegal, Carith.add, Carith.sub, Carith.mul, 
               Carith.illegal, Carith.max, 
               Carith.mulsub, Carith.muladd]
        
        return((OP_F[i])(a, b, c))


# cyclostationary processor class
class Cyclo:
    def __init__(self, fname):
        # set some machine constants
        self.NREGS=256        # number of registers in RF
        self.OPNAME= ['illegal', 'add', 'sub', 'mul', 'illegal', 'max', 
                       'mulsub', 'muladd']
        # has 3 inputs
        self.OPTHREE= [False, False, False, False, False, False, 
                       True, True]

        # initialise the machine state
        self.pc = 0
        self.x = [(i, i * 2 + 1) for i in range(255)]
        self.mem = [int(h, 16) for h in open(fname).readlines()]

    def dump(self):
        return(0)
        # print('pc={}'.format(self.pc))
        # print('reg={}'.format(self.x))
        # print('mem={}'.format(self.mem))

    # decode an instruction to string format
    def decode(self, ir):
        op = (ir >> 29) & 0x7
        rs1 = (ir >> 8) & 0xff
        rs2 = (ir >> 16) & 0xff
        rs3 = (ir >> 24) & 0x1f
        rd = ir & 0xff
        if (self.OPTHREE[op]):
            s = "{} ${},${},${},${}".format(self.OPNAME[op], rd, rs1, rs2, rs3)
        else:
            s = "{} ${},${},${}".format(self.OPNAME[op], rd, rs1, rs2)
        return (op, rs1, rs2, rs3, rd, s)

    # execute one instruction
    def step(self):
        # instruction fetch and decode instruction
        ir = self.mem[self.pc]    
        (op, rs1, rs2, rs3, rd, s) = self.decode(ir)

        # register file
        s1 = self.x[rs1]
        s2 = self.x[rs2]
        if (self.OPTHREE[op]):
            s3 = self.x[rs3]
        else:
            s3 = None
        xold = self.x[rd]

        # execute
        self.x[rd] = Carith.exec(op, s1, s2, s3)

        # display
        print("IF: {} mem[{}]={}".format(s, self.pc, hex(ir)))
        print("x[{}] = {} (was {}) (rs1={},rs2={},rs3={})\n".format(
              rd, self.x[rd], xold, s1, s2, s3))
        self.pc = self.pc + 1


def main(fname):
    c = Cyclo(fname)
    try:
        for i in range(20):
            c.dump()
            c.step()
    except:
        print("Execution completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()
    main(args.fname)

