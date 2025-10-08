import torch
import random
import numpy as np
from npeet import entropy_estimators as ee

def sel_reg(input_dict, n):
    '''
    regularize selection.
    '''
    pa_keys = list(input_dict.keys())
    if not pa_keys: return input_dict
    ch_keys = list(input_dict[pa_keys[0]].keys())

    for s in ch_keys:
        elements = []
        for p in pa_keys: elements.extend(
            [(p, val) for val in input_dict[p][s]]
        )
        total = len(elements)
        if total > n: 
            sampled = random.sample(elements, n)
        else: sampled = elements

        new_lists = {p: [] for p in pa_keys}
        for p, val in sampled: new_lists[p].append(val)
        for p in pa_keys: input_dict[p][s] = new_lists[p]

    return input_dict

def IXZ_est(z, p, no_det= True):
    '''
    estimate the mutual info IXZ straightforwardly.
    '''
    if no_det:
        ps_list = list()
        for k in range(z.size(0)):
            ps_list.append(
                torch.roll(torch.exp(
                    p.log_prob(torch.roll(z, shifts= -k, dims= 0))
                ), shifts= k, dims= 0)
            )
        ps = torch.stack(ps_list)
        log_pz = torch.log(torch.mean(ps, dim= 0))

        return torch.mean(torch.sum((p.log_prob(z) - log_pz), dim= 1))
    else:
        return ee.entropy(z.detach().tolist(), k= 5, base=np.e)

def prod_sampling(p_num, p):
    '''
    sample from p without replacement at p_num times.
    '''
    if p_num.shape != (1,): p_num = p_num.squeeze(0)
    _p_num = torch.distributions.Categorical(p_num)
    K = _p_num.sample()
    log_K = _p_num.log_prob(K)
    p = p.squeeze(0) 

    a = []
    log_p = 0.0
    for _ in range(K+1):
        _p = torch.distributions.Categorical(p)
        a_item = _p.sample()
        a.append(a_item.item())
        
        log_p += _p.log_prob(a_item)
        
        p = p.scatter(-1, a_item.unsqueeze(-1), 0.0)
        p = p/p.sum()

    return a, log_p+log_K

def selection(U, Rxs, Txs, args, dev):
    '''
    sample the total un-regularized selection A from PA|U.
    '''
    u = U.sample()
    u = u.to(dev)

    a = {i:{j:[] for j in range(args.num_of_Tx)} for i in range(args.num_of_Rx)}
    log_p = {i:0. for i in range(args.num_of_Rx)}

    for i, Rx in enumerate(Rxs):
        a_hat, log_p_hat = Rx.select(u)

        for j in a_hat:
            a_jk, log_p_jk = Txs[j].select(u, Rx.em)
            log_p_hat += log_p_jk
            a[i][j].extend(a_jk)
        log_p[i] = log_p_hat

    a = sel_reg(a, int(min(args.num_of_sel_Tx, args.num_of_sel_Rx*args.num_of_Mx)))

    return a, log_p

def full_participation(args):
    a = {i:{j:[l for l in range(args.num_of_Mx)] for j in range(args.num_of_Tx)} for i in range(args.num_of_Rx)}

    return a

def count_selection(a):
    length = 0
    for d in a.values():
        for v in d.values(): length += len(v)
    return length

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 = X0 / normX
    Y0 = Y0 / normY

    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()
    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c

def cal_mpjpe_pampjpe(preds, gts):
    preds = preds.reshape(-1, 17, 3).cpu().numpy()
    gts = gts.reshape(-1, 17, 3).cpu().numpy()

    N = preds.shape[0]
    num_joints = preds.shape[1]

    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))

    pampjpe = np.zeros([N, num_joints])

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    pampjpe = np.mean(pampjpe)

    return mpjpe, pampjpe

def cal_acc(logits, gts):
    preds = torch.argmax(logits, dim= 1)
    acc = (preds == gts).float().mean().item()

    return acc