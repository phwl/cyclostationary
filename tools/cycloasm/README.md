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
	add $1,$2,$3
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
phwl@vlan-2669-10-17-29-23 src % ./cycloasm.py -e -x test.cyc 
3 20030201
4 20301080
5 60311181
6 E0321282
7 C1808183
11 20020004
12 40020004
13 60020004
14 A0020004
15 C3020104
16 E4030205
phwl@vlan-2669-10-17-29-23 src % ./cyclosim.py a.b
IF: add $1,$2,$3 mem[0]=0x20030201
x[1] = (5, 12) (was (1, 3)) (rs1=(2, 5),rs2=(3, 7),rs3=None)

IF: add $128,$16,$48 mem[1]=0x20301080
x[128] = (64, 130) (was (128, 257)) (rs1=(16, 33),rs2=(48, 97),rs3=None)

IF: mul $129,$17,$49 mem[2]=0x60311181
x[129] = (-2632, 3398) (was (129, 259)) (rs1=(17, 35),rs2=(49, 99),rs3=None)

IF: muladd $130,$18,$50,$0 mem[3]=0xe0321282
x[130] = (-2837, 3669) (was (130, 261)) (rs1=(18, 37),rs2=(50, 101),rs3=(0, 1))

IF: mulsub $131,$129,$128,$1 mem[4]=0xc1808183
x[131] = (-20369, 6372) (was (131, 263)) (rs1=(-2632, 3398),rs2=(64, 130),rs3=(5, 12))

IF: add $4,$0,$2 mem[5]=0x20020004
x[4] = (2, 6) (was (4, 9)) (rs1=(0, 1),rs2=(2, 5),rs3=None)

IF: sub $4,$0,$2 mem[6]=0x40020004
x[4] = (-2, -4) (was (2, 6)) (rs1=(0, 1),rs2=(2, 5),rs3=None)

IF: mul $4,$0,$2 mem[7]=0x60020004
x[4] = (-5, 2) (was (-2, -4)) (rs1=(0, 1),rs2=(2, 5),rs3=None)

IF: max $4,$0,$2 mem[8]=0xa0020004
x[4] = (0, 1) (was (-5, 2)) (rs1=(0, 1),rs2=(2, 5),rs3=None)

IF: mulsub $4,$1,$2,$3 mem[9]=0xc3020104
x[4] = (-53, 42) (was (0, 1)) (rs1=(5, 12),rs2=(2, 5),rs3=(3, 7))

IF: muladd $5,$2,$3,$4 mem[10]=0xe4030205
x[5] = (-82, 71) (was (5, 11)) (rs1=(2, 5),rs2=(3, 7),rs3=(-53, 42))

Execution completed
```
