#
# @author:Don Dennis
# machinecodeconst.py
#
# Constants and variable declaring various
# machine instructions
# Update: 3 March, 2021


class MachineCodeConst:
    # Definition of opcodes used in assembly language instructions
    INSTR_ADD = 'add'
    INSTR_SUB = 'sub'
    INSTR_MUL = 'mul'
    INSTR_MAX = 'max'
    INSTR_MULSUB = 'mulsub'
    INSTR_MULADD = 'muladd'

    # All reserved opcodes
    ALL_INSTR = [INSTR_ADD, INSTR_SUB, INSTR_MUL,
                 INSTR_MULADD, INSTR_MULSUB, INSTR_MAX]
    # All instruction in a type
    INSTR_TYPE_R = [INSTR_ADD, INSTR_SUB, INSTR_MUL,
                    INSTR_MAX]
    # All instruction in a type
    INSTR_TYPE_MAC = [INSTR_MULSUB, INSTR_MULADD]


    # Binary Opcodes
    BOP_ARITH = '0000000'

    # The instruction in each distinct binary opcode
    INSTR_BOP_ARITH = [INSTR_ADD, INSTR_SUB, INSTR_MUL,
                       INSTR_MULADD, INSTR_MULSUB, INSTR_MAX]

    # FUNCT for each instruction type
    OPCODE_ARITH = {
        INSTR_ADD: '001',
        INSTR_SUB: '010',
        INSTR_MUL: '100',
        INSTR_MULADD: '101',
        INSTR_MULSUB: '110',
        INSTR_MAX: '111'
    }

