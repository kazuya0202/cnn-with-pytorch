class GlobalVariables:
    def __init__(self):
        # ===== USER SETTINGS =====
        self.image_size = 200
        self.test_size = 10

        self.image_path = r'./recognition_datasets/Images/'
        self.log_path = r'./recognition_datasets/Logs/'

        self.extensions = ['jpg', 'png']

        self.epoch = 10
        self.minibatch_size = 10

        # ======================
