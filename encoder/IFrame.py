from encoder.Frame import Frame
from encoder.params import EncoderConfig


class IFrame(Frame):
    def __init__(self, curr_frame, prev_frame=None ):
        super().__init__(curr_frame)

    def encode(self, encoder_config: EncoderConfig):
        print(f'encoder_config{self}')
        return IFrame(self.curr_frame, self.prev_frame)
