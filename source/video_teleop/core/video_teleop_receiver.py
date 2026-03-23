"""
Video teleoperation receiver - Phase 2.

This module receives hand/arm pose data via ZeroMQ IPC and visualizes it.
No robot control, no IK, no coordinate transformation (only receive → visualize).
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import zmq
import msgpack


class VideoTeleopReceiver:
    """
    Video teleoperation receiver that receives pose data via IPC and visualizes it.
    
    Responsibilities:
    - Receive IPC messages (ZeroMQ SUB)
    - Parse MessagePack data
    - Print/visualize received data
    - Track communication statistics
    """
    
    def __init__(
        self,
        *,
        zmq_address: str = "ipc:///tmp/shadowhand_teleop_video.ipc",
        print_rate_hz: float = 1.0,
        enable_print: bool = True,
    ) -> None:
        """
        Initialize video teleoperation receiver.
        
        Args:
            zmq_address: ZeroMQ address (IPC or TCP)
            print_rate_hz: Rate at which to print messages (Hz, 0 to disable)
            enable_print: Whether to print received messages
        """
        self.zmq_address = zmq_address
        self.print_rate_hz = float(print_rate_hz)
        self.print_period = 1.0 / self.print_rate_hz if self.print_rate_hz > 0 else float('inf')
        self.enable_print = bool(enable_print)
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.socket.connect(self.zmq_address)
        
        # State
        self.is_running = False
        self.receive_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.message_count = 0
        self.last_sequence = None
        self.sequence_gaps = 0
        self.last_message_time = None
        self.last_print_time = 0.0
        self.latest_message: Optional[dict] = None
        
        # Give publisher time to start (ZeroMQ PUB/SUB slow joiner)
        time.sleep(0.1)
        
        print(f"[VideoTeleopReceiver] Initialized")
        print(f"  ZMQ address: {self.zmq_address}")
        print(f"  Print rate: {self.print_rate_hz} Hz")
        print(f"  Print enabled: {self.enable_print}")
    
    def _print_message(self, message: dict) -> None:
        """Print message in human-readable format."""
        if not self.enable_print:
            return
        
        header = message.get("header", {})
        seq = header.get("sequence", -1)
        timestamp = header.get("timestamp", 0.0)
        
        left_hand = message.get("left_hand", {})
        right_hand = message.get("right_hand", {})
        
        left_detected = left_hand.get("detected", False)
        right_detected = right_hand.get("detected", False)
        
        left_pos = left_hand.get("robot_frame", {}).get("wrist_position")
        right_pos = right_hand.get("robot_frame", {}).get("wrist_position")
        left_ori = left_hand.get("robot_frame", {}).get("wrist_orientation")  # Euler xyz (rad)
        right_ori = right_hand.get("robot_frame", {}).get("wrist_orientation")  # Euler xyz (rad")
        
        left_joints = left_hand.get("hand_joints", {}).get("joint_angles")
        right_joints = right_hand.get("hand_joints", {}).get("joint_angles")
        
        # Calculate latency (if timestamp is available)
        latency = None
        if timestamp > 0:
            latency = time.time() - timestamp
        
        print(f"\n{'='*80}")
        print(f"[Message #{seq}] Timestamp: {timestamp:.6f}")
        if latency is not None:
            print(f"  Latency: {latency*1000:.2f} ms")
        
        print(f"\n  Left Hand:")
        print(f"    Detected: {left_detected}")
        if left_detected and left_pos is not None:
            print(f"    Wrist Position (robot frame): [{left_pos[0]:.4f}, {left_pos[1]:.4f}, {left_pos[2]:.4f}] m")
        if left_detected and left_ori is not None and len(left_ori) == 3:
            print(f"    Wrist Rotation (Euler xyz, robot frame): [{left_ori[0]:.4f}, {left_ori[1]:.4f}, {left_ori[2]:.4f}] rad")
        if left_detected and left_joints is not None:
            print(f"    Joint angles: {len(left_joints)} DoF")
            if len(left_joints) > 0:
                print(f"      First 5: {[f'{x:.3f}' for x in left_joints[:5]]}")
        
        print(f"\n  Right Hand:")
        print(f"    Detected: {right_detected}")
        if right_detected and right_pos is not None:
            print(f"    Wrist Position (robot frame): [{right_pos[0]:.4f}, {right_pos[1]:.4f}, {right_pos[2]:.4f}] m")
        if right_detected and right_ori is not None and len(right_ori) == 3:
            print(f"    Wrist Rotation (Euler xyz, robot frame): [{right_ori[0]:.4f}, {right_ori[1]:.4f}, {right_ori[2]:.4f}] rad")
        if right_detected and right_joints is not None:
            print(f"    Joint angles: {len(right_joints)} DoF")
            if len(right_joints) > 0:
                print(f"      First 5: {[f'{x:.3f}' for x in right_joints[:5]]}")
        
        print(f"{'='*80}\n")
    
    def _print_stats(self) -> None:
        """Print communication statistics."""
        elapsed = time.time() - self.last_print_time if self.last_print_time > 0 else 0.0
        if elapsed > 0:
            rate = self.message_count / elapsed
        else:
            rate = 0.0
        
        print(f"\n[Stats] Messages: {self.message_count}, Rate: {rate:.1f} Hz, "
              f"Sequence gaps: {self.sequence_gaps}")
    
    def _receive_loop(self) -> None:
        """Main receive loop (runs in separate thread)."""
        print(f"[VideoTeleopReceiver] Receive loop started")
        
        self.last_print_time = time.time()
        
        while self.is_running:
            try:
                # Non-blocking receive
                try:
                    packed = self.socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                
                # Unpack message
                message = msgpack.unpackb(packed, raw=False)
                self.latest_message = message
                
                # Update statistics
                self.message_count += 1
                header = message.get("header", {})
                seq = header.get("sequence", -1)
                
                if self.last_sequence is not None and seq != self.last_sequence + 1:
                    gap = seq - self.last_sequence - 1
                    self.sequence_gaps += gap
                    print(f"[WARNING] Sequence gap detected: {self.last_sequence} -> {seq} (gap: {gap})")
                
                self.last_sequence = seq
                self.last_message_time = time.time()
                
                # Print message (rate-limited)
                current_time = time.time()
                if current_time - self.last_print_time >= self.print_period:
                    self._print_message(message)
                    self._print_stats()
                    self.last_print_time = current_time
                
            except Exception as e:
                print(f"[ERROR] Receive loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Brief pause before retry
    
    def get_latest_message(self) -> Optional[dict]:
        """Get the latest received message (thread-safe read)."""
        return self.latest_message
    
    def get_stats(self) -> dict:
        """Get communication statistics."""
        return {
            "message_count": self.message_count,
            "last_sequence": self.last_sequence,
            "sequence_gaps": self.sequence_gaps,
            "last_message_time": self.last_message_time,
        }
    
    def start(self) -> None:
        """Start receiving loop."""
        if self.is_running:
            print("[WARNING] Receiver already running")
            return
        
        self.is_running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        print("[VideoTeleopReceiver] Started")
    
    def stop(self) -> None:
        """Stop receiving loop."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.receive_thread is not None:
            self.receive_thread.join(timeout=2.0)
        self.socket.close()
        self.context.term()
        print("[VideoTeleopReceiver] Stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        """Context manager exit."""
        self.stop()

