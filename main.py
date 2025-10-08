import torch
import torch.nn.functional as F
import logging
from datetime import datetime

from model import data, environ, modules
from config import parser


# ----------------- pre-process ----------------- 
# load configs.
args = parser.parse_args()
args.no_conv = True if args.net_en[0] != 'MobileNet' else False
if args.no_det == True and args.sampling == 'opt': args.method = 'pom2dib'
elif args.no_det == True and args.sampling == 'rand': args.method = 'rsdib'
elif args.no_det == True and args.sampling == 'full': args.method = 'tadib'
elif args.no_det == False and args.sampling == 'full': args.method = 'dlsc'
else: raise ValueError(f"invalid baseline: det={args.no_det} + sampling={args.sampling}")

torch.manual_seed(args.seed)

# init the dataset and additional arguements.
D = getattr(data, args.dataset)(args.num_of_sel_Tx, args.num_of_sel_Rx)
for k, v in D.info().items(): setattr(args, k, v)
dev = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# log outputs.
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = 'logs/' + args.dataset + f'_{args.method}_lr_sel_{args.lr_sel}_beta_{args.beta}'\
    + f'_dim_{args.embd_dim}_sel_Tx_{args.num_of_sel_Tx}_Rx_{args.num_of_sel_Rx}_Spr_{args.sparse}_{current_time}.log'
logger = logging.getLogger(log_filename)
format_str = logging.Formatter(
    '%(asctime)s - %(levelname)s : %(message)s'
)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_filename)
stream_handler.setFormatter(format_str)
file_handler.setFormatter(format_str)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# init transmitters and receivers.
Txs = [modules.Transmitter(
    dims= [args.input_dim[idx_Tx], args.embd_dim], net= args.net_en, lr= [args.lr_code, args.lr_sel], 
    num_of_Mx= args.num_of_Mx, no_det= args.no_det
) for idx_Tx in range(args.num_of_Tx)]
Rxs = [modules.Receiver(
    dims= [args.embd_dim, args.output_dim[idx_Rx]], net= args.net_de, lr= [args.lr_code, args.lr_sel], 
    num_of_Tx= args.num_of_Tx, num_of_sel= args.num_of_sel_Rx, 
    num_of_Mxs= int(args.num_of_Mx * args.num_of_Tx)
) for idx_Rx in range(args.num_of_Rx)]

# init common randomness and selection.
U = torch.distributions.Normal(
    torch.tensor([0. for _ in range(args.embd_dim)]), 
    torch.tensor([1. for _ in range(args.embd_dim)])
)
a = [] # total selection.

# ----------------- MAIN ----------------- 
for count in range(args.epochs):
    # init info.
    info = {'loss': 0., 'hyz': 0., 'rate': 0., 'logl': 0.}

    # ----------------- TRAIN ----------------- 
    d = iter(D.load(bs= args.bs))

    for _ in range(args.bs):
        # OPERATION 1: select modalities for receivers' tasks.
        if args.sampling == 'opt':
            a, log_p = environ.selection(U, Rxs, Txs, args, dev)
        elif args.sampling == 'rand': 
            if not a: a, _ = environ.selection(U, Rxs, Txs, args, dev)
        elif args.sampling == 'full': 
            a = environ.full_participation(args)

        # OPERATION 2: feed-forward flow -- transmit compressed z.
        x, y = D.custom(next(d), no_conv= args.no_conv)

        for idx_Rx, Rx in enumerate(Rxs):
            z = list(); loss_loc = 0.; rate = 0.   
            ATTR = 'cross_entropy' if idx_Rx != 0 or \
                args.dataset != 'mmfi' else 'mse_loss'
            for idx_Tx, Tx in enumerate(Txs):
                for idx_Mx in range(args.num_of_Mx):
                    if idx_Mx in a[idx_Rx][idx_Tx]:
                        _z, _p = Tx.send(x[idx_Tx][idx_Mx].to(dev), idx_Mx, Rx.em, args.no_conv)
                        IXZ = environ.IXZ_est(_z, _p, args.no_det)
                        loss_loc += getattr(F, ATTR)(
                            Rx.infer_loc(_z, Tx.em[idx_Mx]), y[idx_Rx].squeeze().to(dev)
                        ) + IXZ
                        rate += IXZ.cpu().detach().item() if args.no_det else IXZ
                    else: _z = torch.zeros((args.bs, args.embd_dim)).to(dev)
                    z.append(_z)
                    
            z = torch.cat(z, dim=1)

            #[NOTE] This is the penality that we introduced in Section VII.A
            pos = 0 if z.ne(0).any().item() else 1

            # OPERATION 3: gradient-backward.
            loss_hyz = getattr(F, ATTR)(Rx.infer(z+pos*1e-6*torch.clamp(torch.randn(
                (args.bs, int(args.num_of_Mx*args.num_of_Tx*args.embd_dim))
            ), -1, 1).to(dev)), y[idx_Rx].squeeze().to(dev))
            loss = loss_hyz + args.beta*loss_loc
            loss.backward()

            if args.sampling == 'opt':
                loss_reg = loss.detach().item()+pos-args.mi[idx_Rx]
                if args.sparse:
                    loss_reg += args.gamma*environ.count_selection(a)
                logl = log_p[idx_Rx]*loss_reg 
                logl.backward()

                info['logl'] += logl.cpu().detach().item()    
            info['loss'] += loss.cpu().detach().item()
            info['hyz'] += loss_hyz.cpu().detach().item()
            info['rate'] += rate

    # OPERATION 4: update all the encoders, decoders, and selectors.
    for Tx in Txs: Tx.update()
    for Rx in Rxs: Rx.update()

    # OPERATION 5: update info.
    for k in info.keys(): info[k] /= args.bs

    # OPERATION 6: log info.
    if count % args.log_freq == 0: 
        # ----------------- TEST ----------------- 
        with torch.no_grad():
            x, y = D.custom(next(iter(D.load(bs= args.bs, train= False))), no_conv= args.no_conv)
            for idx_Rx, Rx in enumerate(Rxs):
                z = list()
                for idx_Tx, Tx in enumerate(Txs):
                    for idx_Mx in range(args.num_of_Mx):
                        if idx_Mx in a[idx_Rx][idx_Tx]: 
                            _z, _ = Tx.send(x[idx_Tx][idx_Mx].to(dev), idx_Mx, Rx.em, args.no_conv)
                        else: _z = torch.zeros((args.bs, args.embd_dim)).to(dev)
                        z.append(_z)
                z = torch.cat(z, dim=1)

                preds = Rx.infer(z)
                ATTR = 'cal_acc' if idx_Rx != 0 or \
                    args.dataset != 'mmfi' else 'cal_mpjpe_pampjpe'
                info[f'{ATTR}_{idx_Rx}'] = getattr(environ, ATTR)(preds, y[idx_Rx].squeeze().to(dev))

        logger.info(f"current {count} iter info: {info}, total selection: {a}")
