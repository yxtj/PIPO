from dataclasses import dataclass


def _show_time_(t):
    if t < 1e-3:
        return f"{t*1e6:.3f} us"
    elif t < 1:
        return f"{t*1e3:.3f} ms"
    else:
        return f"{t:.3f} s"
    
def _show_byte_(b):
    if b < 1e6:
        return f"{b/1e3:.3f} kB"
    elif b < 1e9:
        return f"{b/1e6:.3f} MB"
    else:
        return f"{b/1e9:.3f} GB"

@dataclass
class Stat:
    time_offline: float = 0.0
    byte_offline_send: float = 0.0
    byte_offline_recv: float = 0.0
    time_offline_send: float = 0.0
    time_offline_recv: float = 0.0
    time_offline_comp: float = 0.0
    time_online: float = 0.0
    byte_online_send: float = 0.0
    byte_online_recv: float = 0.0
    time_online_send: float = 0.0
    time_online_recv: float = 0.0
    time_online_comp: float = 0.0
    
    def __add__(self, other):
        return Stat(
            self.time_offline + other.time_offline,
            self.byte_offline_send + other.byte_offline_send,
            self.byte_offline_recv + other.byte_offline_recv,
            self.time_offline_send + other.time_offline_send,
            self.time_offline_recv + other.time_offline_recv,
            self.time_offline_comp + other.time_offline_comp,
            self.time_online + other.time_online,
            self.byte_online_send + other.byte_online_send,
            self.byte_online_recv + other.byte_online_recv,
            self.time_online_send + other.time_online_send,
            self.time_online_recv + other.time_online_recv,
            self.time_online_comp + other.time_online_comp,
        )

    def show(self, promot:str='', n:int=1):
        print("{}: Offline time {}: send {} in {}, recv {} in {}, comp {}. "\
              "Online time {}: send {} in {}, recv {} in {}, comp {}".format(
            promot, _show_time_(self.time_offline),
            _show_byte_(self.byte_offline_send), _show_time_(self.time_offline_send),
            _show_byte_(self.byte_offline_recv), _show_time_(self.time_offline_recv),
            _show_time_(self.time_offline_comp),
            _show_time_(self.time_online/n),
            _show_byte_(self.byte_online_send/n), _show_time_(self.time_online_send/n),
            _show_byte_(self.byte_online_recv/n), _show_time_(self.time_online_recv/n),
            _show_time_(self.time_online_comp/n),
        ))

    def show_offline(self, promot:str=''):
        print("{}: Offline time {}: send {} in {}, recv {} in {}, comp {}".format(
            promot, _show_time_(self.time_offline),
            _show_byte_(self.byte_offline_send), _show_time_(self.time_offline_send),
            _show_byte_(self.byte_offline_recv), _show_time_(self.time_offline_recv),
            _show_time_(self.time_offline_comp),
        ))

    def show_online(self, promot:str='', n:int=1):
        print("{}: Online time {}: send {} in {}, recv {} in {}, comp {}".format(
            promot, _show_time_(self.time_online/n),
            _show_byte_(self.byte_online_send/n), _show_time_(self.time_online_send/n),
            _show_byte_(self.byte_online_recv/n), _show_time_(self.time_online_recv/n),
            _show_time_(self.time_online_comp/n),
        ))

    def show_simple(self, promot:str='', n:int=1):
        print("{}: Offline time {}: send {}, recv {}. Online time {}: send {}, recv {}".format(
            promot, _show_time_(self.time_offline),
            _show_byte_(self.byte_offline_send),
            _show_byte_(self.byte_offline_recv),
            _show_time_(self.time_online/n),
            _show_byte_(self.byte_online_send/n),
            _show_byte_(self.byte_online_recv/n),
        ))
