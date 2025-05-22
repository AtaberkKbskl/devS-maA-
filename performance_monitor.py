import cv2
class PerformanceMonitor:
    """
    FPS ve gecikme ölçümü için yardımcı sınıf.
    """
    def __init__(self):
        self.frame_count = 0
        self.start_time = cv2.getTickCount()

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        end_time = cv2.getTickCount()
        elapsed = (end_time - self.start_time) / cv2.getTickFrequency()
        return self.frame_count / elapsed if elapsed > 0 else 0