from train import train


def test_train():
    train()
    from infer import test_inference
    test_inference()