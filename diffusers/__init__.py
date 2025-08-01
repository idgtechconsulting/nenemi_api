class DiffusionPipeline:
    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None):
        return cls()

    def to(self, device):
        return self

    def __call__(self, prompt):
        class R:
            images = [None]
        return R()
