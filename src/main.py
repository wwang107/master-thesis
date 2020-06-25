from config.defualt import get_cfg_defaults
from data.build import make_dataloader


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.freeze()
    print(cfg)

    data_loader = make_dataloader(cfg, is_train=True)
    iter(data_loader).next()
    pass