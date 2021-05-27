import torch
from openpifpaf import decoder

def get_model(checkpoint) :
    '''
    @param checkpoint: checkpoint file to the model in .pickle
    #TODO: Get this to woprk with urls also
    '''
    checkpoint = torch.load(checkpoint)
    net_cpu = checkpoint['model']
    # initialise for eval and removes stuff for training like dropout layers
    net_cpu.eval()
    model = net_cpu.to('cuda:0')
    processor = decoder.factory(net_cpu.head_metas)
    return model, processor
