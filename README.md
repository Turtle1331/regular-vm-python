# regular-vm-python
Python emulator for the regular-vm ISA: https://github.com/regular-vm/specification

To get started: `$ ./interpreter.py -g cat.hex`

More details: 
```
$ ./interpreter.py -h
usage: Emulate a program with the regular-vm ISA.  [-h] [-x | -m | -g]
                                                   [-s SAVE_MEMORY]
                                                   file

positional arguments:
  file                  the program file

optional arguments:
  -h, --help            show this help message and exit
  -x, --hex             parse program as hex
  -m, --memory          parse program as memory file
  -g, --guess-format    guess program format from file extension
  -s SAVE_MEMORY, --save-memory SAVE_MEMORY
                        save memory to file when stopped
```
