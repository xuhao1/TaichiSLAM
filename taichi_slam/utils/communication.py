import lcm
from .Buffer import Buffer
import random

CHANNEL_SUBMAP = "SUBMAP_CHANNEL"
CHANNEL_TRAJ = "TRAJ_CHANNEL"
TIMEOUT_MS = 10

class SLAMComm:
    def __init__(self, drone_id = 0, lcm_url="udpm://224.0.0.251:7667?ttl=1"):
        self.lcm = lcm.LCM(lcm_url)
        self.drone_id = drone_id
        self.sent_msgs = set()
        self.submap_sub = self.lcm.subscribe(CHANNEL_SUBMAP, self.handle_submap)
        self.traj_sub = self.lcm.subscribe(CHANNEL_TRAJ, self.handle_traj)
    
    def publishBuffer(self, buf, channel=CHANNEL_SUBMAP):
        msg = Buffer()
        msg.drone_id = self.drone_id
        #Generate random hash for msg_id
        msg.msg_id = random.randint(0, 2**16)
        msg.msg_len = len(buf)
        msg.buffer = buf
        self.sent_msgs.add(msg.msg_id)
        self.lcm.publish(channel, msg.encode())
        # print(f"Sent message on channel {channel} msg_id {msg.msg_id} size {len(msg.buffer)/1024:.1f} KB")
    
    def handle_submap(self, channel, data):
        msg = Buffer.decode(data)
        if msg.msg_id in self.sent_msgs:
            return
        # print(f"Received message on channel {channel} msg_id {msg.msg_id}")
        self.on_submap(msg.buffer)

    def handle_traj(self, channel, data):
        msg = Buffer.decode(data)
        if msg.msg_id in self.sent_msgs:
            return
        # print(f"Received message on channel {channel} msg_id {msg.msg_id}")
        self.on_traj(msg.buffer)

    def handle(self):
        self.lcm.handle_timeout(TIMEOUT_MS)