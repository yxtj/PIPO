from dataclasses import dataclass

@dataclass
class Stat:
    time_offline: float = 0.0
    byte_offline_send: float = 0.0
    byte_offline_recv: float = 0.0
    # time_offline_send: float = 0.0
    # time_offline_recv: float = 0.0
    time_online: float = 0.0
    byte_online_send: float = 0.0
    byte_online_recv: float = 0.0
    # time_online_send: float = 0.0
    # time_online_recv: float = 0.0
    
    def __add__(self, other):
        return Stat(
            self.time_offline + other.time_offline,
            self.byte_offline_send + other.byte_offline_send,
            self.byte_offline_recv + other.byte_offline_recv,
            self.time_online + other.time_online,
            self.byte_online_send + other.byte_online_send,
            self.byte_online_recv + other.byte_online_recv,
        )
    
    def show(self, promot:str='', n:int=1):
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
        
        print("{}: Offline time {}, send {}, recv {}; Online time {}, send {}, recv {}".format(
            promot, _show_time_(self.time_offline),
            _show_byte_(self.byte_offline_send), _show_byte_(self.byte_offline_recv),
            _show_time_(self.time_online/n),
            _show_byte_(self.byte_online_send/n), _show_byte_(self.byte_online_recv/n),
        ))

