# https://pysource.com
import pyrealsense2 as rs
import numpy as np


class RealsenseCamera:
    def __init__(self):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.color_frame = None
        self.depth = None

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth = aligned_frames.get_depth_frame()

        color_frame = aligned_frames.get_color_frame()

        self.color_frame = color_frame
        self.depth = depth
        
        if not depth or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None
        
        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)
        
        # Create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return True, color_image, depth_image


    def get_distance_point(self, depth_frame, x, y):
        # Get the distance of a point in the image
        distance = self.depth.get_distance(x, y)
        # convert to cm
        return round(distance * 100, 2)


    
    def release(self):
        self.pipeline.stop()

        


