import cv2 
import traceback 
import threading 
import queue


class JetCamera():
    def __init__(self, cap_w, cap_h, cap_fps):
        #self.cap_orig_w, self.cap_orig_h = 3264, 2464 # 4/3 , 21 fps
        self.cap_orig_w, self.cap_orig_h = 1920, 1080 # 16/9 , 30 fps
        #self.cap_orig_w, self.cap_orig_h = 1280, 720  # 60/120 fps
        self.cap_orig_fps = 30
        self.cap_out_w = cap_w
        self.cap_out_h = cap_h 
        self.cap_out_fps = cap_fps
        self.h_thread = None
        self.b_exit = None
        self.max_queue = 3
        self.queue = queue.Queue(maxsize=self.max_queue)
        
        self.cap_str = 'nvarguscamerasrc tnr-strength=1 tnr-mode=2 ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 '\
                '! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx '\
                '! videorate ! video/x-raw, framerate=(fraction)%d/1 '\
                '! videoconvert !  video/x-raw, format=BGR !  appsink sync=false ' \
                % (self.cap_orig_w, self.cap_orig_h, self.cap_orig_fps, self.cap_out_w, self.cap_out_h, self.cap_out_fps)

        self.cap = None 

    def open(self):
        if self.cap:
            return True 
        try:
            self.cap = cv2.VideoCapture(self.cap_str, cv2.CAP_GSTREAMER)
        except:
            traceback.print_exc()

        self.h_thread = threading.Thread(target=self.read_run)
        self.h_thread.start()

        return self.cap is not None 

    def read_run(self):
        while not self.b_exit:
            try:
                ret, img = self.cap.read()
                if ret:
                    if self.queue.qsize() < self.max_queue:
                        self.queue.put_nowait(img)
            except:
                traceback.print_exc()

    def read(self):
        if not self.cap:
            return False, None

        try:
            img = self.queue.get(block=True, timeout=5)
            if img is None:
                return False, None 
            return True, img
        except:
            pass

        return False, None

    def close(self):

        self.b_exit = True
        try:
            if self.cap:
                self.cap.release()
            self.cap = None 
        except:
            pass 

        try:
            self.queue.put_nowait(None)
        except:
            pass

        self.h_thread.join()
        self.h_thread = None
