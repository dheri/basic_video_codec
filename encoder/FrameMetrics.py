from typing import List


class FrameMetrics:
    def __init__(self, idx: int, is_i_frame: bool, avg_mae: float, mae_comps: int,
                 psnr: float, frame_bytes: int, file_bits: int, encoding_time: float, elapsed_time: float):
        self.idx = idx
        self.is_i_frame = is_i_frame
        self.avg_mae = avg_mae
        self.mae_comps = mae_comps
        self.psnr = psnr
        self.frame_bytes = frame_bytes
        self.file_bits = file_bits
        self.encoding_time = encoding_time
        self.elapsed_time = elapsed_time

    def to_csv_row(self) -> List:
        """Convert the metrics to a list suitable for writing to a CSV row."""
        return [
            self.idx,
            1 if self.is_i_frame else 0,  # Represent I-Frame as 1 and P-Frame as 0
            f"{self.avg_mae:.2f}",
            self.mae_comps,
            f"{self.psnr:.2f}",
            self.frame_bytes,
            self.file_bits,
            f"{self.encoding_time:.2f}",
            f"{self.elapsed_time:.2f}",
        ]

    @staticmethod
    def from_csv_row(row: List) -> 'FrameMetrics':
        """Create a FrameMetrics instance from a CSV row."""
        return FrameMetrics(
            idx=int(row[0]),
            is_i_frame=bool(int(row[1])),
            avg_mae=float(row[2]),
            mae_comps=int(row[3]),
            psnr=float(row[4]),
            frame_bytes=int(row[5]),
            file_bits=int(row[6]),
            encoding_time=float(row[7]),
            elapsed_time=float(row[8])
        )

    @staticmethod
    def get_header():
        return ["idx", "I-Frame", "avg_MAE", "mae_comps", "PSNR", "frame_bytes", "file_bits", "enc_time",
                "elapsed_time"]

    def __repr__(self):
        return (f"FrameMetrics(idx={self.idx}, is_i_frame={self.is_i_frame}, avg_mae={self.avg_mae:.2f}, mae_comps="
                f"{self.mae_comps}, psnr={self.psnr:.2f}, frame_bytes={self.frame_bytes}, file_bits="
                f"{self.file_bits}), encoding_time={self.encoding_time:.2f}, elapsed_time={self.elapsed_time:.2f}")
