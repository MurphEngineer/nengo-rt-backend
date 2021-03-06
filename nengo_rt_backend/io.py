import socket
from time import sleep
import logging
import sys
import struct
import math
import fcntl

log = logging.getLogger(__name__)

class IOController(object):
    def __init__(self):
        pass

    def recv(self):
        """Receive decoded values from the target. Returns a list of floating-point DVs."""
        raise NotImplementedError("IOController must implement recv()")

class EthernetIOController(IOController):

    def __init__(self, mac_address, device):
        IOController.__init__(self)
        self.dst_addr_str = mac_address
        self.dst_addr = []
        rest = mac_address
        while len(rest) > 0:
            first = rest[0:2]
            rest = rest[2:]
            self.dst_addr.append(int(first, 16))
        self.dst_addr = bytes(self.dst_addr)
        self.target_ethertype = bytes([136, 181]) # 0x88b5 experimental ethertype 1
        self.recv_opcode = bytes([1]) # 0x01
        self.device = device
        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
        self.sock.bind((self.device, 0))
        self.src_addr = self.get_mac_address()

    # FIXME duplicate code from controller.py, consider separating into another file

    def get_mac_address(self):
        # FIXME this may not be portable to non-POSIX systems
        info = fcntl.ioctl(self.sock.fileno(), 0x8927, struct.pack('256s', 
                                                           bytearray(self.device[:15], 'UTF-8')
                                                           ))
        return info[18:24]

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
    
    def makeframe_SEND(self, addr, count, data):
        return self.makeframe(bytes([1]), addr + count + data) # 0x01

    def float2int(self, f):
        if f < 0.0:
            sgn = True
            f = f * -1
        else:
            sgn = False
        # clamp to representable range
        if f > 2047.0 / 1024.0:
            f = 2047.0 / 1024.0
        ipart = int(math.floor(f))
        fpart = int(round((f - ipart) * 2**10))
        val = ipart * 2**10 + fpart
        if sgn:
            val = 2**12 - val
        return val

    def pack2(self, x1, x2):
        # Pack 2 floating-point values into 3 octets using 12-bit sfixed(2, 10) representation.
        i1 = self.float2int(x1)
        i2 = self.float2int(x2)
        # first byte is the highest 8 bits of i1
        # second byte is the lowest 4 bits of i1, then the highest 4 bits of i2
        # third byte is the lowest 8 bits of i2
        return bytes([i1>>4, (i1&0x0f)<<4 | i2>>8, i2&0x0ff])
    
    def send(self, dvData):
        if len(dvData) == 0:
            return
        pairCount = int(math.ceil(len(dvData)/2))
        if pairCount > 256:
            raise ValueError("sending more than 256 pairs of input values, handle this case")
        address = bytes([0, 0])
        pairOctet = bytes([pairCount - 1])
        data = b""
        for i in range(pairCount):
            x1 = dvData[2 * i]
            if len(dvData) > 2*i+1:
                x2 = dvData[2 * i + 1]
            else:
                x2 = 0.0
            data += self.pack2(x1, x2)
        self.sock.send(self.makeframe_SEND(address, pairOctet, data))

    def recv(self):
        received = False
        frame = None
        received_dvs = []
        while not received:
# FIXME we need to know ahead of time how many DVs/DV pairs to read -> how many frames to recv
            frame = self.sock.recv(1518) # Ethernet frame without FCS
            ethernet_header = frame[:14]
            # check source address and ethertype
            src_addr = ethernet_header[6:12]
            ethertype = ethernet_header[12:14]
            if src_addr == self.dst_addr and ethertype == self.target_ethertype:
                # check payload for correct opcode
                ethernet_payload = frame[14:]
                # payload format:
                # 1 octet tag
                # 1 octet opcode (should be 0x01)
                # 2 octet start address
                # 1 octet pair count
                # then (pair count + 1) worth of 6 octet DV pairs
                opcode = ethernet_payload[1:2]
                if opcode == self.recv_opcode:
                    received = True
                    start_addr = ethernet_payload[2:4] # not currently used
                    pair_count = ethernet_payload[4:5]
                    # ...to integer
                    pair_count = pair_count[0] + 1
                    pair_data = ethernet_payload[5:]
                    for i in range(pair_count):
                        pair = pair_data[0:3]
                        pair_data = pair_data[3:]
                        # each pair is two 12-bit values packed into 3 bytes
                        # first unpack each DV as an integer
                        dv0 = 2**4 * pair[0] + (pair[1] >> 4)
                        dv1 = 2**8 * (pair[1] & 0x0f) + pair[2]
                        # now perform 2's complement
                        if dv0 > 2047:
                            dv0 = -1 * (2**12 - dv0)
                        if dv1 > 2047:
                            dv1 = -1 * (2**12 - dv1)
                        # convert to sfixed
                        dv0 *= 2**-10
                        dv1 *= 2**-10
                        # and append to data collection array
                        received_dvs.append(dv0)
                        received_dvs.append(dv1)
        return received_dvs
