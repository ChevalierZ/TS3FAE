import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SDLoss(nn.Module):
    def __init__(self, size_average=True):
        super(SDLoss, self).__init__()
        self.size_average = size_average

    def forward(self, output, label, target_spectral, flag):
        loss_func = torch.nn.MSELoss(size_average=self.size_average) # linalg.vector_norm(output-label,ord=2)
        loss = loss_func(output,label)
        loss = loss
        return loss.view(-1, 1)

    # def sam(self,r, d):
    #     rr = torch.pow(torch.diag(torch.mm(r, torch.transpose(r, 0, 1))), 0.5)
    #     dd = torch.pow(torch.diag(torch.mm(d, torch.transpose(d, 0, 1))), 0.5)
    #     rd = torch.abs(torch.mm(r, torch.transpose(d, 0, 1)))
    #     result = rd / torch.mul(rr, dd)
    #     return result

    # def sid(self, p_1, p_2):
    #     r = p_2
    #     d = p_1
    #     m = torch.div(r, torch.sum(r,0))
    #     n = torch.div(d, torch.sum(d,0))
    #     drd = torch.sum(torch.mul(m, torch.log(torch.div(m, n+(1e-30)))), 0)
    #     ddr = torch.sum(torch.mul(n, torch.log(torch.div(n, m+(1e-30)))), 0)
    #     result = drd + ddr
    #     return result