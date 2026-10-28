"""Synthesized car audio for the simulator: electric motor whine + tire slip squeal.

Motor pitch follows speed, loudness follows |acceleration|.
Tire squeal loudness follows the slip angle atan2(vy, |vx|).
Plays through ALSA (libasound) via ctypes, so no extra Python dependencies.
"""
import ctypes
import math
import sys
import threading

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

SR = 44100
BLOCK = 1024  # ~23 ms per write


class Synth:
    def __init__(self):
        self.motor_phase = 0.0
        self.squeal_phase = 0.0
        self.last = (0.0, 0.0, 0.0)  # (motor freq, motor amp, tire amp)

    def block(self, speed, accel, slip):
        speed, accel, slip = abs(speed), abs(accel), abs(slip)
        motor_f = 40.0 + 75.0 * speed
        motor_a = 0.03 + 0.07 * min(speed / 10.0, 1.0) + 0.2 * min(accel / 6.0, 1.0)
        # no squeal near standstill, where the slip angle is ill-defined
        tire_a = 0.35 * np.clip((slip - 0.05) / 0.25, 0.0, 1.0) * min(speed / 2.0, 1.0)

        # ramp every parameter across the block to avoid clicks
        f0, ma0, ta0 = self.last
        f = np.linspace(f0, motor_f, BLOCK)
        ma = np.linspace(ma0, motor_a, BLOCK)
        ta = np.linspace(ta0, tire_a, BLOCK)
        self.last = (motor_f, motor_a, tire_a)

        ph = self.motor_phase + 2 * np.pi * np.cumsum(f) / SR
        self.motor_phase = ph[-1] % (2 * np.pi)
        motor = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)

        squeal_f = 900.0 + 150.0 * np.random.randn(BLOCK).cumsum() / np.sqrt(BLOCK)
        sph = self.squeal_phase + 2 * np.pi * np.cumsum(squeal_f) / SR
        self.squeal_phase = sph[-1] % (2 * np.pi)
        noise = np.convolve(np.random.randn(BLOCK), np.ones(6) / 6, 'same')
        tire = 0.6 * np.sin(sph) + noise

        out = np.clip(ma * motor / 1.6 + ta * tire, -1.0, 1.0)
        return (out * 32767).astype(np.int16)


def pick_device():
    """Ask on the terminal which ALSA playback device to use; 'default' if there is no terminal."""
    devices = [('default', 'ALSA default')]
    with open('/proc/asound/pcm') as f:  # lines like "01-00: USB Audio : USB Audio : playback 1 : capture 1"
        for line in f:
            if 'playback' in line:
                card, dev = line.split(':')[0].split('-')
                devices.append((f'plughw:{int(card)},{int(dev)}', line.split(':')[1].strip()))
    if not sys.stdin.isatty():
        return 'default'
    for i, (name, label) in enumerate(devices, 1):
        print(f'{i}: {label} ({name})')
    while True:
        choice = input(f'Select audio device [1-{len(devices)}]: ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(devices):
            return devices[int(choice) - 1][0]


class SimAudio(Node):
    def __init__(self):
        super().__init__('sim_audio')
        device = self.declare_parameter('alsa_device', '').value or pick_device()
        self.speed = self.accel = self.slip = 0.0
        self.prev = None

        self.alsa = ctypes.CDLL('libasound.so.2')
        self.alsa.snd_pcm_writei.restype = ctypes.c_long
        # without argtypes ctypes passes the buffer address as a 32-bit int and segfaults
        self.alsa.snd_pcm_writei.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        self.alsa.snd_strerror.restype = ctypes.c_char_p
        self.pcm = ctypes.c_void_p()
        # 0 = SND_PCM_STREAM_PLAYBACK, 2 = S16_LE, 3 = RW_INTERLEAVED, 1 channel, 100 ms latency
        err = self.alsa.snd_pcm_open(ctypes.byref(self.pcm), device.encode(), 0, 0)
        err = err or self.alsa.snd_pcm_set_params(self.pcm, 2, 3, 1, SR, 1, 100000)
        if err < 0:
            raise RuntimeError(f'ALSA device {device!r}: {self.alsa.snd_strerror(err)}')

        self.create_subscription(Odometry, '/state_estimation/odom', self.odom_cb, 10)
        threading.Thread(target=self.audio_loop, daemon=True).start()
        self.get_logger().info(f'Playing on ALSA device {device!r}, listening to /state_estimation/odom')

    def odom_cb(self, msg):
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        speed = math.hypot(vx, vy)
        if self.prev is not None and t > self.prev[0]:
            a = (speed - self.prev[1]) / (t - self.prev[0])
            self.accel += 0.1 * (a - self.accel)  # low-pass the finite difference
        self.prev = (t, speed)
        self.speed = speed
        self.slip = math.atan2(vy, abs(vx)) if speed > 0.1 else 0.0

    def audio_loop(self):
        synth = Synth()
        while rclpy.ok():
            buf = synth.block(self.speed, self.accel, self.slip)
            n = self.alsa.snd_pcm_writei(self.pcm, buf.ctypes.data, BLOCK)
            if n < 0:  # underrun etc.
                self.alsa.snd_pcm_recover(self.pcm, ctypes.c_int(n), 1)


def main(args=None):
    rclpy.init(args=args)
    node = SimAudio()
    rclpy.spin(node)
    node.alsa.snd_pcm_close(node.pcm)
    rclpy.shutdown()
