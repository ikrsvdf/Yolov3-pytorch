import os
import datetime
from torch.utils.tensorboard import SummaryWriter

'''
   用来将监控数据写入文件系统（日志），保存训练的某些信息。如损失等。
   这个logger类在train.py中使用，在训练过程中保存一些信息到日志文件。
'''

class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:  # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):#将监控数据写入日志
        """Log a scalar variable."""

        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):#将监控数据批量写入日志
        """Log scalar variables."""

        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()
