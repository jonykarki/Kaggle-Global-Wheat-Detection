from .WheatModel import WheatModel

def build_model(cfg):
    model = WheatModel(cfg)
    return model