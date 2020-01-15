class GlobalVariables:
    def __init__(self):
        # ===== USER SETTINGS =====
        self.image_size = 3
        self.test_size = 1
        self.train_size = self.image_size - self.test_size

        self.image_path = r'./Images/'
        self.log_path = r'./Logs/'

        self.extensions = ['jpg', 'png']

        # ======================
