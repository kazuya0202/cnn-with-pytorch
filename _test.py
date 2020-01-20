import datetime
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader

# my packages
import torch_utils as tu
import utils as ul
from global_variables import GlobalVariables


class Test:
    def __init__(self, path, pt_path):
        self.path = Path(path)
        self.pt_path = pt_path

        # get file name
        self.filename_base = str(datetime.datetime.now().strftime(
            "ymd%Y%m%d_hms%H%M%S")).replace(" ", "_")

    def execute(self):
        gv = GlobalVariables()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_size = gv.image_size
        if not isinstance(image_size, tuple):
            image_size = (image_size, image_size)

        # dataset
        """ MEMO
        classes
            - dataset.classes は使わない
            - ul.load_classes を使う

        """

        test_data = DataLoader()

        # parameters for LogFile
        log_params = [gv.log_path, self.filename_base]
        p = None if not gv.is_save_debug_log \
            else ul.create_file_path(*log_params, head='test-')
        test_log_file = ul.LogFile(p)

        model = tu.Model(device, image_size, load_path=self.pt_path)
        test_model = tu.TestModel(model, test_data, logs=test_log_file)


if __name__ == "__main__":
    image_path = r'./recognition_datasets/Images/'
    pt_path = r'./recognition_datasets/model.pt'

    test = Test(image_path, pt_path)
    test.execute()
