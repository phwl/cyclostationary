# Introduction
A simple assembler based on https://github.com/metastableB/RISCV-RV32I-Assembler.

# Installation
First make sure you have all the required libraries installed.`
```console
pip3 install -r requirements.txt
```

Then follow the usage instructions at <https://github.com/metastableB/RISCV-RV32I-Assembler>, e.g.
```console
phwl@phwlnuc:~/src/cyclostationary/tools/cycloasm/src$ cat test.cyc 
START: 
	# Louis' examples
	mul $128,$1,$0
	mul $129,$3,$2
	mul $130,$5,$4
	add $129,$128,$2

mid:
	# every opcode
	add $4,$0,$2
	sub $4,$0,$2
	mul $4,$0,$2
	max $4,$0,$2
	mulsub $4,$0,$2
	muladd $4,$0,$2
phwl@phwlnuc:~/src/cyclostationary/tools/cycloasm/src$ ./cycloasm.py -x -es test.cyc 
Symbols and Addresses:
{'START': 0, 'mid': 16}
phwl@phwlnuc:~/src/cyclostationary/tools/cycloasm/src$ cat a.b 
0B800001
0B810203
0B820405
09810280
09040200
0A040200
0B040200
0D040200
0E040200
0F040200

```
