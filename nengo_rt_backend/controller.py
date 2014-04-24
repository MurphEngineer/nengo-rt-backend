from socket import socket, AF_PACKET, SOCK_RAW
from time import sleep
import logging
import sys
import struct
import math
import fcntl

log = logging.getLogger(__name__)

class Controller(object):
    
    def __init__(self, loadfile):
        self.loadfile = loadfile

    def reset(self):
        raise NotImplementedError("Controller must implement reset()")

    def program(self):
        raise NotImplementedError("Controller must implement program()")

    def start(self):
        raise NotImplementedError("Controller must implement start()")

    def pause(self):
        raise NotImplementedError("Controller must implement pause()")
    
    def step(self):
        raise NotImplementedError("Controller must implement step()")

class EthernetController(Controller):

    def __init__(self, loadfile, mac_address, device):
        Controller.__init__(self, loadfile)
        self.dst_addr_str = mac_address # as a string
        # similar to binstring() but the reverse operation and for hex bytes
        self.dst_addr = []
        rest = mac_address
        while len(rest) > 0:
            first = rest[0:2]
            rest = rest[2:]
            self.dst_addr.append(int(first, 16))
        self.dst_addr = bytes(self.dst_addr)
        self.device = device
        self.sock = socket(AF_PACKET, SOCK_RAW)
        self.sock.bind((self.device, 0))

    def reset(self):
        self.sock.send(self.makeframe_RESET())
        sleep(0.1)

    def start(self):
        self.sock.send(self.makeframe_START())

    def pause(self):
        self.sock.send(self.makeframe_PAUSE())

    def step(self):
        self.sock.send(self.makeframe_STEP())

    def binstring(self, s):
        retval = []
        rest = s
        while(len(rest) > 0):
            first = rest[0:8]
            rest = rest[8:]
            retval.append(int(first, 2))
        return bytes(retval)

    def makeframe(self, opcode, data):
        ethertype = b"\x88\xb5"
        tag = b"\x42" # FIXME better tag handling, especially when the board learns how to respond
        frame = b"".join([self.dst_addr, self.src_addr, ethertype, tag, opcode, data])
        # padding
        while len(frame) < 60:
            frame = b"".join([frame, b'\x00'])
        return frame

    def makeframe_RESET(self):
        return self.makeframe(bytes([255]), b"") # 0xff
    def makeframe_START(self):
        return self.makeframe(bytes([254]), b"") # 0xfe
    def makeframe_STEP(self):
        return self.makeframe(bytes([253]), b"") # 0xfd
    def makeframe_PAUSE(self):
        return self.makeframe(bytes([252]), b"") # 0xfc

    def program(self):
        log.debug("programming " + self.dst_addr_str + " via " + self.device)
        # FIXME this may not be portable to non-POSIX systems
        info = fcntl.ioctl(self.sock.fileno(), 0x8927, struct.pack('256s', 
                                                           bytearray(self.device[:15], 'UTF-8')
                                                           ))
        self.src_addr = info[18:24]
        self.src_addr_str = ''.join(['%02x' % char for char in info[18:24]])
        log.debug("local MAC address on device is " + self.src_addr_str)        
        # reset
        self.reset()
        # send loadfile (but do not start running the simulation!)
        log.info("transmitting loadfile")
        f = open(self.loadfile, 'r')
        for line in f:
            # FIXME input validation
            tmp = line.split()
            # binary characters -> binary digits
            addr = tmp[0]
            data = tmp[1]
            # opcode 0x00 = programming data
            self.sock.send(self.makeframe(b'\x00', self.binstring(addr) + self.binstring(data)))
        f.close()
        log.info("finished sending loadfile")
