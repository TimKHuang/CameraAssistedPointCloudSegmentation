import os


class PredWriter:

    def __init__(self, path):
        self.scan_index = 0
        self.seq_index = 0

        if path is None:
            self.path = os.path.join(os.path.abspath(os.path.curdir), "result")
        else:
            self.path = path
    
    def init_path(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.path = os.path.join(self.path, "sequences")
        if not os.path.exists(self.path):
            os.mkdir(self.path)
    
    def switch_sequence(self, sequence, scan=0):
        self.scan_index = scan
        self.seq_index = sequence

        seq_path = os.path.join(self.path, f"{self.seq_index:02d}")
        if not os.path.exists(seq_path):
            os.mkdir(seq_path)
        
        pred_path = os.path.join(seq_path, "predictions")
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
    
    def write(self, pred):
        fpath = os.path.join(
            self.path, 
            f"{self.seq_index:02d}", 
            "predictions",
            f"{self.scan_index:06d}.label"
        )
        pred.tofile(fpath)
        self.scan_index += 1
        print("File " + fpath + " written.")