import torch
from model import functionals
from model.environ import prod_sampling as ps

dev = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


class Transmitter(object):
    def __init__(
            self, dims= [], net= ['Base', 'Base'], lr= [1e-4, 0.5e-4], num_of_Mx= int(), no_det= True
        ):
        self.no_det = no_det
        self.em = [1e-1*torch.randn(dims[-1]).to(dev) for _ in range(num_of_Mx)]
        
        self._init_ens_and_sel(dims, net, num_of_Mx)
        self._init_opts(lr)

        self._init_Mxs_rec()

    def _init_ens_and_sel(self, dims, net, num_of_Mx):
        input_dims, em_dim = dims
        ATTR = 'Encoder' if self.no_det else 'EncoderDet'
        self.ens= [
            getattr(functionals, ATTR)(
                [int(input_dims[i]+em_dim), em_dim], net[0]
            ) for i in range(num_of_Mx)
        ]
        self.sel= getattr(functionals, 'Selector')(
            [int(2*em_dim), num_of_Mx, num_of_Mx], net[1]
        )

    def _init_opts(self, lr= [1e-4, 0.5e-4]):
        self.opts= []
        for en in self.ens: self.opts.append(
            torch.optim.AdamW(en.net.parameters(),lr= lr[0], weight_decay= 1e-4)
        )
        self.opts.append(
            torch.optim.AdamW(self.sel.net.parameters(),lr= lr[1], weight_decay= 1e-4)
        )
          
    def _init_Mxs_rec(self):
        self.Mxs = []

    def update(self):
        for i in self.Mxs:
            self.opts[i].step()
            self.opts[i].zero_grad()
        
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.sel.net.parameters(), 10
        )
        self.opts[-1].step()
        self.opts[-1].zero_grad()

        self._init_Mxs_rec()
        
    def send(self, x, idx_Mx, em_Rx, no_conv= True):
        em_Rx = em_Rx.unsqueeze(0)
        em_Rx_repeated = em_Rx.repeat(x.size(0), 1)

        if no_conv: p = self.ens[idx_Mx].encode(
                torch.cat((x, em_Rx_repeated), dim= 1)
            )
        else: p = self.ens[idx_Mx].encode(
               (x, em_Rx_repeated)
            )
        z = p.rsample() if self.no_det else p
        
        if idx_Mx not in self.Mxs:
            self.Mxs.append(idx_Mx)

        return z, p
    
    def select(self, u, em_Rx):
        p_num, p = self.sel.select(torch.cat((u, em_Rx)))
        a, log_p = ps(p_num, p)

        return a, log_p


class Receiver(object):
    def __init__(
            self, dims= [], net= ['Base', 'Base'], lr= [1e-4, 0.5e-4],
            num_of_Tx= int(), num_of_sel= int(), num_of_Mxs= int()
        ):
        self.em = 1e-1*torch.randn(dims[0]).to(dev)

        self.init_des_and_sel(dims, net, num_of_Tx, num_of_Mxs, num_of_sel)
        self.init_opt(lr)
        
    def init_des_and_sel(self, dims, net, num_of_Tx, num_of_Mxs, num_of_sel):
        em_dim, output_dim= dims
        self.de= getattr(functionals, 'Decoder')(
            [int(num_of_Mxs*em_dim), output_dim], net[0]
        )
        self.de_loc= getattr(functionals, 'Decoder')(
            [int(2*em_dim), output_dim], net[0]
        )
        self.sel= getattr(functionals, 'Selector')(
            [em_dim, num_of_Tx, num_of_sel], net[1]
        )

    def init_opt(self, lr= [1e-4, 0.5e-4]):
        self.opt= torch.optim.AdamW([
            {'params': self.de.net.parameters(), 'lr': lr[0], 'weight_decay': 1e-4},
            {'params': self.de_loc.net.parameters(), 'lr': lr[0], 'weight_decay': 1e-4},
            {'params': self.sel.net.parameters(),'lr': lr[1], 'weight_decay': 1e-4},
        ])

    def update(self):
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.sel.net.parameters(), 10
        )
        self.opt.step()
        self.opt.zero_grad()

    def infer(self, z):
        return self.de.decode(z)
    
    def infer_loc(self, z, em_Mx):
        em_Mx = em_Mx.unsqueeze(0)
        em_Mx_repeated = em_Mx.repeat(z.size(0), 1)

        z_loc = torch.cat((z, em_Mx_repeated), dim= 1)
        
        return self.de_loc.decode(z_loc)
    
    def select(self, u):
        p_num, p = self.sel.select(u)
        a, log_p = ps(p_num, p)

        return a, log_p
