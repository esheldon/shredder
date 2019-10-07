class PSFFailure(Exception):
    """
    failure to bootstrap PSF
    """
    def __init__(self, value):
        super(PSFFailure, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


