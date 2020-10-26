# Introduction
A simple assembler based on https://github.com/metastableB/RISCV-RV32I-Assembler.

# Installation
First make sure you have all the required libraries installed.`
```console
pip3 install -r requirements.txt
```

Then follow the usage instructions at <https://github.com/metastableB/RISCV-RV32I-Assembler>, e.g.
```console
phwl@vlan-2669-10-17-29-23 src % cat test.cyc              
START: 
	# Louis' examples
	add $128,$16,$48
	mul $129,$17,$49
	muladd $130,$18,$50,$0
	mulsub $131,$129,$128,$1

mid:
	# every opcode
	add $4,$0,$2
	sub $4,$0,$2
	mul $4,$0,$2
	max $4,$0,$2
	mulsub $4,$1,$2,$3
	muladd $5,$2,$3,$4
phwl@vlan-2669-10-17-29-23 src % ./cycloasm.py -x -e test.cyc 
3 20301080
4 60311181
5 E0321282
6 C1808183
10 20020004
11 40020004
12 60020004
13 A0020004
14 C3020104
15 E4030205
phwl@vlan-2669-10-17-29-23 src % cat a.b
20301080
60311181
E0321282
C1808183
20020004
40020004
60020004
A0020004
C3020104
E4030205
```
