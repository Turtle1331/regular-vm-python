#! /usr/bin/env python2
import math
from enum import Enum
import struct
import sys
import argparse
import time

try:
	import console
	print ''
	console.clear()
except ImportError:
	pass

'''

32 registers
2^32 bytes (4 MiB) of addressable memory
Program loaded starting at address 0
mainloop:
	read instruction
	increment pc
	execute instruction
	repeat

'''


DEBUG = False
DEBUG_STEP_MODE = True

ERROR_ON_BAD_DEVICE = True


def int_to_bytes(data, data_len):
	data %= 2 ** (data_len * 8)
	if data_len == 1:
		return chr(data)
	elif data_len == 2:
		return struct.pack('<H', data)
	elif data_len == 4:
		return struct.pack('<L', data)
	else:
		raise ValueError('Unrecognized data length')

def bytes_to_int(data, signed):
	data_len = len(data)
	if data_len == 1:
		return ord(data)
	elif data_len == 2:
		return struct.unpack('<h' if signed else '<H', data)[0]
	elif data_len == 4:
		return struct.unpack('<l' if signed else '<L', data)[0]
	else:
		raise ValueError('Unrecognized data length')



class OffsetPolicy (Enum):
	'''Policies for how to handle read/write requests at misaligned offsets'''
	RaiseError = 0   # Raise a ValueError
	WordAlign = 1   # Align to the previous word boundary
	WriteOffset = 2   # Write at misaligned offset


WORD_SIZE = 4   # 32-bit words


class MemoryUnit (object):
	PAGE_SIZE = 2**10   # 1 KiB per page
	PAGE_COUNT = 2**22   # 4 MiB total
	def __init__(self, offset_write_policy=OffsetPolicy.RaiseError, offset_read_policy=OffsetPolicy.RaiseError):
		self.offset_write_policy = offset_write_policy
		self.offset_read_policy = offset_read_policy
		self.pages = {}
		self.prog_size = 0   # Number of pages taken up by program

		self._memory_map = {}


	def load_program(self, program):
		# Delete all pages occupied by an existing program
		for index in range(self.prog_size):
			self.delete_page(index)

		# Load the new program
		self.prog_size = (len(program) - 1) // self.PAGE_SIZE + 1
		for index in range(self.prog_size):
			self.create_page(index, program[index * self.PAGE_SIZE:], overwrite=True)

		if DEBUG:
			print 'load program into', self.prog_size, ('page' if self.prog_size == 1 else 'pages')


	def create_page(self, index, data=None, overwrite=False):
		index = int(index)
		data = data or bytearray()
		if index in self.pages and not overwrite:
			raise RuntimeError('Attempted to overwrite page without overwrite flag')
		if index < 0 or index >= self.PAGE_COUNT:
			raise IndexError('Page index out of bounds')

		if DEBUG:
			print 'create page at index', hex(index), 'with', len(data), 'bytes of data'
		page = bytearray(self.PAGE_SIZE)
		if data:
			data_len = min(self.PAGE_SIZE, len(data))
			page[:data_len] = data[:data_len]
		self.pages[index] = page


	def delete_page(self, index):
		del self.pages[index]


	def map_memory(self, device, start_addr, data_len):
		if DEBUG:
			print 'Map memory from', hex(start_addr), 'to', hex(start_addr + data_len - 1), 'to device', device.__class__.__name__
		for offset in range(data_len):
			self._memory_map[start_addr + offset] = (device, offset)


	def write_data(self, addr, data, device=True):
		if device and self._memory_map and addr in self._memory_map:
			device, offset = self._memory_map[addr]
			dev_len = device.DEVICE_LENGTH
			data = data[:dev_len - offset]
			if DEBUG:
				print 'write to device', device.__class__.__name__, 'at offset', offset, ':', repr(str(data))
			device.write_data(offset, data)
		else:
			self._write_memory(addr, data)


	def read_data(self, addr, data_len, device=True):
		if device and self._memory_map and addr in self._memory_map:
			device, offset = self._memory_map[addr]
			dev_len = device.DEVICE_LENGTH
			data_len = min(data_len, dev_len - offset)
			if DEBUG:
				print 'read from device', device.__class__.__name__, 'at offset', offset, ':'
			value = device.read_data(offset, data_len)
			if DEBUG:
				print 'read from device: data is', repr(str(value))
			if not isinstance(value, bytearray):
				raise TypeError('Device did not return bytearray')
			return value
		else:
			return self._read_memory(addr, data_len)


	def _write_memory(self, addr, data):
		data_len = len(data)
		if data_len > WORD_SIZE:
			raise ValueError('Attempted to write more than one word')

		addr = int(addr)
		old_addr = addr
		index = addr // self.PAGE_SIZE
		addr &= (self.PAGE_SIZE - 1)

		if data_len > 1 and addr % WORD_SIZE != 0:
			policy = self.offset_write_policy
			if policy == OffsetPolicy.RaiseError:
				raise ValueError('Attempted to write at misaligned offset')
			elif policy == OffsetPolicy.WordAlign:
				addr -= addr % 4
			elif policy == OffsetPolicy.WriteOffset:
				extra = addr + data_len - self.PAGE_SIZE
				if extra > 0:
					data_len -= extra
					self.write_data(index + addr, data[:data_len], device=False)
					self.write_data(index + addr + extra, data[data_len:], device=False)

		if index not in self.pages:
			self.create_page(index)
		page = self.pages[index]

		if DEBUG:
			print 'write data at page', hex(index), ', address', hex(addr), 'to', repr(data), '(was', repr(page[addr:addr + data_len]), ')'
		page[addr:addr + data_len] = data


	def _read_memory(self, addr, data_len):
		#if data_len > WORD_SIZE:
			#raise ValueError('Attempted to read more than one word')

		addr = int(addr)
		old_addr = addr
		index = addr // self.PAGE_SIZE
		addr &= (self.PAGE_SIZE - 1)

		if data_len > 1 and addr % WORD_SIZE != 0:
			policy = self.offset_read_policy
			if policy == OffsetPolicy.RaiseError:
				raise ValueError('Attempted to read at misaligned offset')
			elif policy == OffsetPolicy.WordAlign:
				addr -= addr % WORD_SIZE
			elif policy == OffsetPolicy.WriteOffset:
				extra = addr + data_len - self.PAGE_SIZE
				if extra > 0:
					data_len -= extra
					return self.read_data(index + addr, data[:data_len], device=False) + self.read_data(index + addr + extra, data[data_len:], device=False)

		if index not in self.pages:
			return bytearray(data_len)
		page = self.pages[index]
		if DEBUG:
			print 'read data at page', hex(index), ', address', hex(addr), ':', repr(str(page[addr:addr + data_len]))
		return page[addr:addr + data_len]


	def save_to_file(self, fname):
		with open(fname, 'wb') as f:
			# Write magic string with binary safegaurd (4 bytes)
			f.write('\xffMEM')

			# Write page count (4 bytes)
			f.write(int_to_bytes(len(self.pages), 4))

			for index, page in self.pages.items():
				# Write page index (4 bytes)
				f.write(int_to_bytes(index, 4))
				# Write page data (1 KiB)
				f.write(page)

			# Write file ending (3 bytes!)
			f.write('OK\x00')


	def load_from_file(self, fname):
		with open(fname, 'rb') as f:
			# Read magic string (4 bytes)
			magic = f.read(4)
			if magic != '\xffMEM':
				raise SyntaxError('Memory file is missing magic string')

			# Read page count (4 bytes)
			page_count = f.read(4)
			if len(page_count) < 4:
					raise SyntaxError('Memory file contains incomplete page count')

			page_count = bytes_to_int(page_count, False)

			pages = {}
			for i in range(page_count):
				# Read page index (4 bytes)
				index = f.read(4)
				if len(index) < 4:
					raise SyntaxError('Memory file contains incomplete index')

				index = bytes_to_int(index, 4)

				# Read page data (1 KiB)
				page = f.read(self.PAGE_SIZE)
				if len(page) < self.PAGE_SIZE:
					raise SyntaxError('Memory file contains incomplete page')

				pages[index] = bytearray(page)

			ending = f.read(4)
			end_len = len(ending)
			if end_len < 3:
				raise SyntaxError('Memory file is shorter than expected')
			elif end_len > 3:
				raise SyntaxError('Memory file is longer than expected')
			elif ending != 'OK\x00':
				raise SyntaxError('Memory file contains bad ending')

		self.pages = pages



class RegisterUnit (object):
	REG_COUNT = 32   # 32 registers
	def __init__(self, wrap_addr=False):
		self.wrap_addr = wrap_addr
		self.registers = None
		self.reset()


	def get_value(self, addr):
		if addr < 0 or addr >= self.REG_COUNT:
			if self.wrap_addr:
				addr %= self.REG_COUNT
			else:
				raise IndexError('register address out of range')

		if DEBUG:
			print 'get value at', addr, ':', repr(str(int_to_bytes(self.registers[addr], 4)))
		return self.registers[addr]


	def set_value(self, addr, value):
		addr = int(addr)
		if addr < 0 or addr >= self.REG_COUNT:
			if self.wrap_addr:
				addr %= self.REG_COUNT
			else:
				raise IndexError('register address out of range')


		if (addr == 0):
			# Wrap program counter as unsigned value
			value %= 2 ** (WORD_SIZE * 8)
		else:
			# Wrap as signed value
			half = 2 ** (WORD_SIZE * 8 - 1)
			value = (value + half) % (2 * half) - half

		if DEBUG:
			print 'set value at', addr, 'to', repr(str(int_to_bytes(value, 4))), '(was', repr(str(int_to_bytes(self.registers[addr], 4))), ')'
		self.registers[addr] = value


	def get_pc(self):
		pc = self.registers[0] % (2 ** (WORD_SIZE * 8 - 1))
		if DEBUG:
			print 'pc is', hex(pc)
		return pc


	def increment_pc(self):
		self.registers[0] = (self.registers[0] + WORD_SIZE) % (2 ** (WORD_SIZE * 8 - 1))


	def reset(self):
		self.registers = [0] * 32



class InstructionUnit (object):
	def __init__(self, register_unit, memory_unit, send_func):
		self.instructions = {
			0x00: self.nop,
			0x01: self.add,
			0x02: self.sub,
			0x03: self.and_,
			0x04: self.orr,
			0x05: self.xor,
			0x06: self.not_,
			0x07: self.lsh,
			0x08: self.ash,
			0x09: self.tcu,
			0x0a: self.tcs,
			0x0b: self.set,
			0x0c: self.mov,
			0x0d: self.ldw,
			0x0e: self.stw,
			0x0f: self.ldb,
			0x10: self.stb,
			0xfd: self.snd,
			0xff: self.hlt,
		}
		self.registers = register_unit
		self.memory = memory_unit
		self.send_func = send_func


	def execute_instruction(self, word):
		opcode = word[0]
		word = word[1:]

		try:
			inst = self.instructions[opcode]
		except KeyError:
			raise RuntimeError('Unsupported instruction: %r' % opcode)
		name = inst.__name__.rstrip('_')
		tokens = inst.__defaults__ or ()
		args = []

		for token in tokens:
			token_type = token[:-1]
			token_length = int(token[-1])

			arg = word[:token_length]
			word = word[token_length:]
			if token_type == 'r':
				arg = bytes_to_int(arg, False)
			elif token_type == 'imm':
				arg = bytes_to_int(arg, True)
			else:
				raise ValueError('Unrecognized token instruction type')
			args.append(arg)

		if DEBUG:
			print '>', name, ' '.join(map(str, args))
		inst(*args)


	def three_reg_op(func):
		def inst(self, ra='r1', rb='r1', rc='r1'):
			vb = self.registers.get_value(rb)
			vc = self.registers.get_value(rc)
			va = func(self, vb, vc)
			self.registers.set_value(ra, va)
		inst.__name__ = func.__name__
		return inst

	def nop(self):
		pass

	@three_reg_op
	def add(self, x, y):
		return x + y


	@three_reg_op
	def sub(self, x, y):
		return x - y


	@three_reg_op
	def and_(self, x, y):
		return x & y


	@three_reg_op
	def orr(self, x, y):
		return x | y


	@three_reg_op
	def xor(self, x, y):
		return x ^ y


	def not_(self, ra='r1', rb='r1'):
		vb = self.registers.get_value(rb)
		va = ~vb
		self.registers.set_value(ra, va)


	@three_reg_op
	def lsh(self, x, y):
		width = WORD_SIZE * 8
		x &= 2 ** width - 1
		y = max(-width, min(width, y))
		if y > 0:
			return x << y
		else:
			return x >> -y


	@three_reg_op
	def ash(self, x, y):
		width = WORD_SIZE * 8
		y = max(-width, min(width, y))
		if y > 0:
			return x << y
		else:
			return x >> -y


	@three_reg_op
	def tcu(self, x, y):
		width = WORD_SIZE * 8
		x &= 2 ** width - 1
		y &= 2 ** width - 1
		return 1 if x - y > 0 else (0 if x == y else -1)


	@three_reg_op
	def tcs(self, x, y):
		return 1 if x - y > 0 else (0 if x == y else -1)


	def set(self, ra='r1', imm='imm2'):
		self.registers.set_value(ra, imm)


	def mov(self, ra='r1', rb='r1'):
		v = self.registers.get_value(rb)
		self.registers.set_value(ra, v)


	def ldw(self, ra='r1', rb='r1'):
		mb = self.registers.get_value(rb)
		mb &= (2 ** (WORD_SIZE * 8)) - 1
		vb = self.memory.read_data(mb, 4)
		va = bytes_to_int(vb, True)
		self.registers.set_value(ra, va)


	def stw(self, ra='r1', rb='r1'):
		vb = self.registers.get_value(rb)
		va = int_to_bytes(vb, 4)
		ma = self.registers.get_value(ra)
		ma &= (2 ** (WORD_SIZE * 8)) - 1
		self.memory.write_data(ma, va)


	def ldb(self, ra='r1', rb='r1'):
		mb = self.registers.get_value(rb)
		mb &= (2 ** (WORD_SIZE * 8)) - 1
		vb = self.memory.read_data(mb, 1)
		vb = bytes_to_int(vb, False)
		va = self.registers.get_value(ra)
		va = (va << 8) | vb
		self.registers.set_value(ra, va)


	def stb(self, ra='r1', rb='r1'):
		vb = self.registers.get_value(rb)
		va = int_to_bytes(vb, 1)
		ma = self.registers.get_value(ra)
		ma &= (2 ** (WORD_SIZE * 8)) - 1
		self.memory.write_data(ma, va)


	def snd(self, ra='r1', rb='r1'):
		vb = self.registers.get_value(rb)
		va = self.registers.get_value(ra)
		va &= (2 ** (WORD_SIZE * 8)) - 1
		self.send_func(va, vb)


	def hlt(self, ra='r1'):
		va = self.registers.get_value(ra)
		print 'Halt with code 0x%x.' % va
		sys.exit()


class DeviceBase (object):
	def __init__(self):
		self.device_interface = None

	def attach(self, device_interface):
		self.device_interface = device_interface

	def receive(self, data):
		raise NotImplemented()

	def detach(self):
		self.device_interface = None


class TerminalInputDevice (DeviceBase):
	def __init__(self):
		super(TerminalInputDevice, self).__init__()
		self.input_prefix = 'Terminal input: ' if DEBUG else '> '
		self.input_buffer = bytearray()

	def receive(self, addr):
		if not self.input_buffer:
			try:
				if DEBUG and DEBUG_STEP_MODE:
					print self.input_prefix,
					time.sleep(1)
					inp = raw_input()
				else:
					inp = raw_input(self.input_prefix)
			except EOFError:
				raise KeyboardInterrupt()

			inp = inp.encode('ascii', 'ignore')
			self.input_buffer = bytearray(inp)
			self.input_buffer.append(0x00)

		value = self.input_buffer[:1]
		self.input_buffer.pop(0)
		self.device_interface.write_bytes(addr, value, 1)



class TerminalOutputDevice (DeviceBase):
	def __init__(self):
		super(TerminalOutputDevice, self).__init__()
		self.output_prefix = 'Terminal output: ' if DEBUG else ''
		self.chunk_size = 1024

	def receive(self, addr):
		# addr &= ~3
		output = bytearray(self.output_prefix)
		
		while True:
			chunk = self.device_interface.read_bytes(addr, self.chunk_size)
			
			end_pos = chunk.find('\x00')
			if end_pos == -1:
				output += chunk
			else:
				output.extend(chunk[:end_pos])
				break
		
		print output.decode('ascii', 'ignore')



class BusInterface (object):
	def __init__(self, memory_unit):
		self._memory_unit = memory_unit
	
	def read_bytes(self, addr, num_bytes):
		return self._memory_unit.read_data(addr, num_bytes, device=False)
	
	def read_words(self, addr, num_words):
		return self._memory_unit.read_data(addr, num_words * 4, device=False)
	
	def write_bytes(self, addr, data, num_bytes):
		data = data[:num_bytes]
		self._memory_unit.write_data(addr, data)
	
	def write_words(self, addr, data, num_words):
		data = data[:num_words * 4]
		self._memory_unit.write_data(addr, data)


class DeviceInterface (object):
	def __init__(self, bus_interface, device_id):
		self._bus_interface = bus_interface
		self._device_id = device_id

	def read_bytes(self, addr, num_bytes):
		return self._bus_interface.read_bytes(addr, num_bytes)

	def read_words(self, addr, num_words):
		return self._bus_interface.read_words(addr, num_words)

	def write_bytes(self, addr, data, num_bytes):
		self._bus_interface.write_bytes(addr, data, num_bytes)

	def write_words(self, addr, data, num_words):
		self._bus_interface.write_words(addr, data, num_words)

	def device_id(self):
		return self._device_id



class CentralUnit (object):
	def __init__(self, program):
		self.register_unit = RegisterUnit()
		self.memory_unit = MemoryUnit()
		self.bus_interface = BusInterface(self.memory_unit)
		self.instruction_unit = InstructionUnit(self.register_unit, self.memory_unit, self.send_value)
		self.devices = {}

		self.register_device(TerminalInputDevice, 0x01)
		self.register_device(TerminalOutputDevice, 0x02)

		self.memory_unit.load_program(program)


	def mainloop(self):
		try:
			while True:
				pc = self.register_unit.get_pc()
				inst_word = self.memory_unit.read_data(pc, 4, device=False)
				self.register_unit.increment_pc()
				self.instruction_unit.execute_instruction(inst_word)
				if DEBUG and DEBUG_STEP_MODE:
					try:
						if raw_input():
							raise KeyboardInterrupt()
					except EOFError:
						raise KeyboardInterrupt()
		except KeyboardInterrupt as e:
			self.stop()
			raise e


	def stop(self):
		for _, device in self.devices.items():
			device.detach()


	def register_device(self, device_cls, device_id):
		device = device_cls()
		device_inter = DeviceInterface(self.bus_interface, device_id)
		device.attach(device_inter)
		self.devices[device_id] = device


	def send_value(self, val_a, val_b):
		device_id, device_cmd = val_a, val_b
		if device_id in self.devices:
			device = self.devices[device_id]
			device.receive(device_cmd)
		elif ERROR_ON_BAD_DEVICE:
			raise RuntimeError('no device found with id 0x%x' % device_id)

		'''
		val_a &= (2 ** (2 * 8)) - 1
		val_bc = (val_b << (2 * 8)) | val_c
		if val_a == 0x0000:
			# Real mode
			pass
		elif val_a == 0x10:
			# Memory-mapped IO
			start_addr = self.register_unit.get_value(val_b)
			dev_id = val_c
			dev_len = 0
			if dev_id in self.devices:
				device, dev_len = self.devices[dev_id]
				self.memory_unit.map_memory(device, start_addr, dev_len)

			self.register_unit.set_value(val_b, dev_len)
		'''



if __name__ == '__main__':
	parser = argparse.ArgumentParser('Emulate a program with the regular-vm ISA. ')
	parser.add_argument('file', help='the program file')
	file_format = parser.add_mutually_exclusive_group()
	file_format.add_argument('-x', '--hex', action='store_true', help='parse program as hex')
	file_format.add_argument('-m', '--memory', action='store_true', help='parse program as memory file')
	file_format.add_argument('-g', '--guess-format', action='store_true', help='guess program format from file extension')
	parser.add_argument('-w', '--write-program', help='write parsed program to binary file')
	parser.add_argument('-s', '--save-memory', help='save memory to file when stopped')
	parser.add_argument('-d', '--debug', action='store_true', help='emulate program in debug mode')
	args = parser.parse_args()

	if args.guess_format:
		ext = args.file.split('.')[-1]
		if ext == '' or ext == 'rvm':
			# Regular-VM binary file
			pass
		elif ext == 'hex':
			# Hex file
			args.hex = True
		elif ext == 'mem' or ext == 'mems':
			# Memory file
			args.memory = True
			if ext == 'mems':
				# Persistent memory file
				if not args.save_memory:
					args.save_memory = args.file
		else:
			raise RuntimeError('Could not guess program format from extension')

	DEBUG = args.debug

	# Load program file
	if args.hex:
		with open(args.file) as f:
			lines = f.readlines()
		lines = map(lambda x: x.partition('#')[0], lines)
		hexcode = ''.join(lines)
		hexcode = filter(lambda x: x in '0123456789abcdefABCDEF', hexcode)
		program = bytearray.fromhex(hexcode)
	elif args.memory:
		# Program loaded directly as memory
		program = bytearray()
	else:
		with open(args.file, 'rb') as f:
			program = bytearray(f.read())

	if args.write_program:
		with open(args.write_program, 'wb') as f:
			f.write(program)

	cpu = CentralUnit(program)
	if args.memory:
		cpu.memory_unit.load_from_file(args.file)

	try:
		cpu.mainloop()
	except KeyboardInterrupt:
		pass
	finally:
		if args.save_memory:
			cpu.memory_unit.save_to_file(args.save_memory)
