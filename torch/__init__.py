float16 = 'float16'
float32 = 'float32'

class _Cuda:
    @staticmethod
    def is_available():
        return False

cuda = _Cuda()
