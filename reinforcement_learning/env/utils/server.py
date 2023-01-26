import socket
import struct

from env.utils import RECV_LENGTH, SIZE_OF_DATA, BASE_RTT
from collections import namedtuple

from config.config import Config

RawFeatures = namedtuple(
    'RawFeatures',
    [
        'cnps_received', 'rtt_packet_delay', 'monitor_interval_width',
        'nacks_received', 'bytes_sent', 'flow_tag', 'packets_sent',
        'cur_rate', 'host',
    ]
)


class Server:
    """
    Implements the connection between the OMNeTpp simulator and the Python GYM env.
    """
    def __init__(self, config: Config, port: int):
        self.config = config
        self.port = port
        self.server_socket = self.open_server_socket(port=self.port)
        self.server_socket.settimeout(300)
        self.connection, self.client_address = None, None

    def step(self, action) -> RawFeatures:
        """
        Send the action to the simulator, receive and parse the new data.
        :param action: The multiplier by which we change the send rate.
        :return: The parsed raw features provided by the simulator.
        """
        self.send_data(action)
        self.close_connection()
        return self.receive_data()

    def receive_data(self) -> RawFeatures:
        """
        Opens a connection to the OMNeTpp simulator and receives a data stream.

        :return: The raw-features provided by the simulator.
        """
        try:
            data = None
            while not data:
                self.connection, self.client_address = self.server_socket.accept()
                data = self.connection.recv(RECV_LENGTH * SIZE_OF_DATA)
            self.connection.sendall(b'receive status: OK')
            unpacked_data = struct.unpack('I' * RECV_LENGTH, data)
            features = RawFeatures(
                rtt_packet_delay=unpacked_data[0] / BASE_RTT, # rtt latency in nanoseconds normalized by 8192 as the base_rtt ,
                nacks_received=unpacked_data[1], # number nack packets received
                cnps_received=unpacked_data[2] , # number cnp packets received
                bytes_sent=unpacked_data[3], # number of bytes sent 
                cur_rate=unpacked_data[4] * 1. / (1 << 20), # current rate between min_rate and 1 in units of fixed-point 20
                monitor_interval_width=unpacked_data[5], # number nack packets received
                packets_sent=unpacked_data[6], # number of packets sent
                flow_tag=str(unpacked_data[7]), # flowtag
                host=str(unpacked_data[8]), # flowtag
            )
            return features
        except socket.timeout:
            print('socket timed-out')
            return None

    def send_data(self, action) -> None:
        """
        :param action: A multiplier that sets the change in the requested transmission rate.
        """
        try:
            data = struct.pack('I'*1, int(round(action * 1024 * 64)))  # 2 ** 16
            self.connection.sendall(data)
        except socket.timeout:
            print("Timeout Exception couldn't send data")

    def close_connection(self) -> None:
        self.connection.close()

    def end_connection(self) -> None:
        """
        Before closing the connection, send a "close connection request" to the client. This should prevent sockets from
        staying open.
        """
        try:
            self.send_data(1. / (1024 * 64))
            self.connection.close()
        except:
            print("connection already closed from client side")
            pass

    def reset(self) -> RawFeatures:
        try:
            return self.receive_data()
        except:
            print("restarting env")
            self.server_socket = self.open_server_socket(port=self.port)
            self.server_socket.settimeout(300)
            return self.receive_data()

    @staticmethod
    def open_server_socket(port: int, host: str = 'localhost') -> socket.socket:
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(300)
        return serversocket

