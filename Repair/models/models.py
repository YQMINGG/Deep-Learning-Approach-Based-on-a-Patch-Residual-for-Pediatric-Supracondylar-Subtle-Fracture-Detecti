from .MEDFE import MEDFE
def create_model(opt):
    model = MEDFE(opt)
    # print("model [%s] was created" % (model.name()))
    # for name in model.name():
    #     print("{}".format(name))
    return model

