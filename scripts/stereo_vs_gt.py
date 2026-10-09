#!/usr/bin/env python3
"""Compare stereo depth (OpenCV SGBM on the simulated OAK-D Lite pair) with the
ideal Gazebo depth camera, which is pixel-aligned with the LEFT image.

Shows: left image | SGBM depth | ground-truth depth | |error|, and prints
fill rate and depth error statistics.

Requires the bridge from config/oakd_bridge.yaml to be running.
Run:  python3 scripts/stereo_vs_gt.py --ros-args -p use_sim_time:=true
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo

BASELINE = 0.075        # m, OAK-D Lite
MAX_DEPTH = 8.0         # m, colour-map range
NUM_DISP = 96           # multiple of 16; min depth ~ f*B/NUM_DISP


class StereoVsGt(Node):
    def __init__(self):
        super().__init__("stereo_vs_gt")
        self.br = CvBridge()
        self.fx = None
        self.create_subscription(CameraInfo, "/oakd/left/camera_info", self.on_info, 1)
        subs = [Subscriber(self, Image, t) for t in
                ("/oakd/left/image_raw", "/oakd/right/image_raw", "/oakd/depth_gt/image")]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.02)
        self.sync.registerCallback(self.on_frames)
        # Roughly the OAK "default" preset: 7x7 blocks, smoothness penalties.
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISP, blockSize=7,
            P1=8 * 7 * 7, P2=32 * 7 * 7, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def on_info(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.get_logger().info(
                f"fx = {self.fx:.1f} px -> min depth ~ {self.fx * BASELINE / NUM_DISP:.2f} m")

    def on_frames(self, left_msg, right_msg, gt_msg):
        if self.fx is None:
            return
        left = self.br.imgmsg_to_cv2(left_msg, "mono8")
        right = self.br.imgmsg_to_cv2(right_msg, "mono8")
        gt = self.br.imgmsg_to_cv2(gt_msg, "32FC1").astype(np.float32)

        disp = self.sgbm.compute(left, right).astype(np.float32) / 16.0
        valid = disp > 0.5
        depth = np.zeros_like(disp)
        depth[valid] = self.fx * BASELINE / disp[valid]

        gt_ok = np.isfinite(gt) & (gt > 0)
        both = valid & gt_ok
        err = np.abs(depth - gt)
        if both.any():
            e = err[both]
            self.get_logger().info(
                f"fill {100 * both.sum() / gt_ok.sum():5.1f}% | "
                f"median |err| {np.median(e) * 100:5.1f} cm | "
                f"p95 |err| {np.percentile(e, 95) * 100:5.1f} cm | "
                f"median GT {np.median(gt[both]):.2f} m")

        def cmap(d, vmax):
            d8 = (np.clip(np.nan_to_num(d, posinf=0), 0, vmax) / vmax * 255).astype(np.uint8)
            c = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
            c[d8 == 0] = 0
            return c

        err_vis = np.where(both, err, 0)
        row = np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                         cmap(depth, MAX_DEPTH), cmap(np.where(gt_ok, gt, 0), MAX_DEPTH),
                         cmap(err_vis, 0.5)])
        cv2.imshow("left | stereo depth | ground truth | |error| (0-50 cm)", row)
        cv2.waitKey(1)


def main():
    rclpy.init()
    rclpy.spin(StereoVsGt())


if __name__ == "__main__":
    main()
