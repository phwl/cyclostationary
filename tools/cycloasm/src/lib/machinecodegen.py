#
# @author:Don Dennis
# MachineCodeGen.py
#
# Conver the tokenized assembly instruction to
# corresponding machine code

from lib.cprint import cprint as cp
from lib.machinecodeconst import MachineCodeConst


class MachineCodeGenerator:
    CONST = MachineCodeConst()

    def __init__(self):
        '''
        Class that implements the machine code generation part
        for RV32I subset.
        '''
        pass

    def get_bin_register(self, r):
        '''
        converts the register in format
        r'[0-9][0-9]*' to its equivalent
        binary
        '''
        r = r[1:]
        try:
            r = int(r)
        except:
            cp.cprint_fail("Internal Error: get_bin_register:" +
                          " Register could not be parsed")
        assert(r >= 0)
        assert(r < 256)
        rbin = format(r, '08b')
        return rbin

    def op_arith(self, tokens):
        '''
        opcode  rs2 rs1 rd 
        '''
        opcode = tokens['opcode']
        bin_opcode = None
        rs1 = None
        rs2 = None
        rd = None
        bin_rs1 = None
        bin_rs2 = None
        bin_rd = None
        bin_res = '0000'
        bin_wb = '1'

        try:
            bin_opcode = self.CONST.OPCODE_ARITH[opcode]
            rs1 = tokens['rs1']
            rs2 = tokens['rs2']
            rd = tokens['rd']
            bin_rs1 = self.get_bin_register(rs1)
            bin_rs2 = self.get_bin_register(rs2)
            bin_rd = self.get_bin_register(rd)
        except:
            cp.cprint_fail("Internal Error: ARITH: could not parse" +
                           "tokens in " + str(tokens['lineno']))
            exit()

        bin_str = bin_res + bin_wb + bin_opcode + bin_rd + bin_rs2 + bin_rs1
        assert(len(bin_str) == 32)

        tok_dict = {
            'opcode': bin_opcode,
            'rs1': bin_rs1,
            'rd': bin_rd,
            'rs2': bin_rs2
        }
        return bin_str, tok_dict

    def convert_to_binary(self, tokens):
        '''
        The driver function for converting tokens to machine code.
        Takes the tokens parsed by the lexer and returns the
        binary equivalent.

        Returns a touple (instr, dict),
        where instr is the binary string of the instruction
        and the dict is the tokens converted individually
        '''
        try:
            opcode = tokens['opcode']
        except KeyError:
            print("Internal Error: Key not found (opcode)")
            return None

        if opcode in self.CONST.INSTR_BOP_ARITH:
            return self.op_arith(tokens)
        else:
            cp.cprint_fail("Error:" + str(tokens['lineno']) +
                           ": Opcode: '%s' not implemented" % opcode)
            return None

        print("Internal Error: Control should not reach here!")
        return None


mcg = MachineCodeGenerator()
