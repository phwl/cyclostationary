
# parsetab.py
# This file is automatically generated. Do not edit.
_tabversion = '3.8'

_lr_method = 'LALR'

_lr_signature = '66F0D6268AE09CF65140CDBD71EC0D75'
    
_lr_action_items = {'LABEL':([0,10,14,],[3,13,19,]),'OPCODE':([0,],[5,]),'NEWLINE':([0,6,8,12,13,17,18,19,24,],[4,9,-9,15,16,21,22,23,25,]),'$end':([1,2,4,9,15,16,21,22,23,25,],[0,-1,-10,-2,-6,-7,-3,-5,-8,-4,]),'COLUMN':([3,],[6,]),'REGISTER':([5,10,14,20,],[8,8,8,8,]),'COMMA':([7,8,11,17,],[10,-9,14,20,]),'IMMEDIATE':([10,14,],[12,18,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'program':([0,],[1,]),'statement':([0,],[2,]),'register':([5,10,14,20,],[7,11,17,24,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> program","S'",1,None,None,None),
  ('program -> statement','program',1,'p_program_statement','parser.py',34),
  ('program -> LABEL COLUMN NEWLINE','program',3,'p_program_label','parser.py',42),
  ('statement -> OPCODE register COMMA register COMMA register NEWLINE','statement',7,'p_statement_R','parser.py',54),
  ('statement -> OPCODE register COMMA register COMMA register COMMA register NEWLINE','statement',9,'p_statement_MAC','parser.py',70),
  ('statement -> OPCODE register COMMA register COMMA IMMEDIATE NEWLINE','statement',7,'p_statement_I_S_SB','parser.py',86),
  ('statement -> OPCODE register COMMA IMMEDIATE NEWLINE','statement',5,'p_statement_U_UJ','parser.py',131),
  ('statement -> OPCODE register COMMA LABEL NEWLINE','statement',5,'p_statement_UJ_LABEL','parser.py',162),
  ('statement -> OPCODE register COMMA register COMMA LABEL NEWLINE','statement',7,'p_statement_SB__JALR_LABEL','parser.py',178),
  ('register -> REGISTER','register',1,'p_register','parser.py',203),
  ('statement -> NEWLINE','statement',1,'p_statement_none','parser.py',215),
]
