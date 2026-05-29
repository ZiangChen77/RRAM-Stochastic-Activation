import torch
import numpy as np
import matplotlib.pyplot as plt


class Memristor:
    def __init__(self, w, Wmin=-1.0, Wmax=1.0, Gmin=1e-6, Gmax=1e-5, alpha_p=4, alpha_d=4, Np=20, Nd=20, gamma_p=0.5, gamma_d=0.5, gamma_gmin=0.05, gamma_gmax=0.05, mode='lin'):
        self.mode = mode
        if alpha_p == 0.0 and alpha_d == 0.0:
            self.Gmin = Gmin
            self.Gmax = Gmax

        else:
            logGmin = np.log10(Gmin)+torch.randn_like(w).detach()*gamma_gmin
            logGmax = np.log10(Gmax)+torch.randn_like(w).detach()*gamma_gmax
            if self.mode == 'lin':
                self.Gmin = (10**logGmin).to(w.device)
                self.Gmax = (10**logGmax).to(w.device)
            elif self.mode == 'log':
                self.log_Gmin = logGmin
                self.log_Gmax = logGmax
        self.Gscale = (Gmax - Gmin) / (Wmax - Wmin)
        self.Gref = (Gmin + Gmax) / 2
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.alpha_p = alpha_p
        self.alpha_d = alpha_d
        self.Np = Np
        self.Nd = Nd
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.G = w.detach() * self.Gscale + self.Gref
        self.G.clamp_(Gmin, Gmax).to(w.device)

        w.data =(w.data.clamp(Wmin, Wmax)*4.0).fix()/4.0

        if mode == 'log':
            self.log_G = torch.log10(self.G)

    def update_weights(self, dw):
        with torch.no_grad():
            if self.mode == 'lin':
                if self.alpha_p == 0.0 and self.alpha_d == 0.0:
                    dGp = (self.Gmax - self.Gmin) / self.Np
                    self.G[dw > 0] = self.G[dw > 0] + dGp

                    dGd = - (self.Gmax - self.Gmin) / self.Nd
                    self.G[dw < 0] = self.G[dw < 0] + dGd

                else:
                    dGp = ((self.Gmax[dw > 0] - self.Gmin[dw > 0]) / (1 - np.exp(-self.alpha_p)) - (self.G[dw > 0] - self.Gmin[dw > 0])) * (1 - np.exp(-self.alpha_p / self.Np))
                    dGp = dGp * (1 + self.gamma_p * torch.randn_like(dGp).to(dw.device))
                    self.G[dw > 0] = self.G[dw > 0] + dGp

                    dGd = -((self.Gmax[dw < 0] - self.Gmin[dw < 0]) / (1 - np.exp(-self.alpha_d)) - (self.Gmax[dw < 0] - self.G[dw < 0])) * (1 - np.exp(-self.alpha_d / self.Nd))
                    dGd = dGd * (1 + self.gamma_d * torch.randn_like(dGd).to(dw.device))
                    self.G[dw < 0] = self.G[dw < 0] + dGd

                self.G.clamp_(self.Gmin, self.Gmax)

                w = (self.G - self.Gref) / self.Gscale
            elif self.mode == 'log':
                dGp = ((self.log_Gmax[dw > 0] - self.log_Gmin[dw > 0]) / (1 - np.exp(-self.alpha_p)) - (self.log_G[dw > 0] - self.log_Gmin[dw > 0])) * (1 - np.exp(-self.alpha_p / self.Np))
                dGp = dGp * (1 + self.gamma_p * torch.randn_like(dGp).to(dw.device))
                self.log_G[dw > 0] = self.log_G[dw > 0] + dGp

                dGd = -((self.log_Gmax[dw < 0] - self.log_Gmin[dw < 0]) / (1 - np.exp(-self.alpha_d)) - (self.log_Gmax[dw < 0] - self.log_G[dw < 0])) * (1 - np.exp(-self.alpha_d / self.Nd))
                dGd = dGd * (1 + self.gamma_d * torch.randn_like(dGd).to(dw.device))
                self.log_G[dw < 0] = self.log_G[dw < 0] + dGd

                self.log_G.clamp_(self.log_Gmin, self.log_Gmax)

                self.G = 10**self.log_G
            w = (self.G - self.Gref) / self.Gscale
            return w


def ltp_ltd_plot(memristor_params, filename='memristor'):
    w = torch.ones((20,), requires_grad=True) * -1.0
    w.memristor = Memristor(w, **memristor_params)

    w_ltp_ltd = []
    G_ltp_ltd = []
    w_ltp_ltd.append(w.clone().detach().numpy().tolist())
    G_ltp_ltd.append(w.memristor.G.clone().detach().numpy().tolist())
    for _ in range(5):
        for _ in range(w.memristor.Np):
            dw = torch.ones_like(w)
            w.data = w.memristor.update_weights(dw)
            G_ltp_ltd.append(w.memristor.G.clone().detach().numpy().tolist())
            w_ltp_ltd.append(w.clone().detach().numpy().tolist())
        for _ in range(w.memristor.Nd):
            dw = torch.zeros_like(w) - 1.0
            w.data = w.memristor.update_weights(dw)
            G_ltp_ltd.append(w.memristor.G.clone().detach().numpy().tolist())
            w_ltp_ltd.append(w.clone().detach().numpy().tolist())

    w_traces = np.array(w_ltp_ltd).T
    fig = plt.figure()
    for i in range(len(w)-1):
        plt.plot(w_traces[i], marker='o',color='cyan',linewidth=1, markersize=6, alpha=0.5)
    plt.plot(w_traces[-1], marker='o',color='blue',linewidth=2, markersize=8)
    plt.grid()
    plt.xlabel('Pulse Number')
    plt.ylabel('Weight')
    plt.title('Memristor LTP&LTD')
    plt.savefig(filename+'_ltp_ltd_w.svg')
    plt.close()
        
    G_traces = np.array(G_ltp_ltd).T
    fig = plt.figure()
    for i in range(len(w)-1):
        plt.plot(G_traces[i], marker='o',color='cyan',linewidth=1, markersize=6, alpha=0.5)
    plt.plot(G_traces[-1], marker='o',color='blue',linewidth=2, markersize=8)
    plt.grid()
    plt.xlabel('Pulse Number')
    plt.ylabel('Conductance [S]')
    # plt.yscale('log')
    plt.title('Memristor LTP&LTD')
    plt.savefig(filename+'_ltp_ltd_G.svg')
    plt.close()

class OptimizerAdamMem(torch.optim.Adam):
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, threshold=100.0): 
        super(OptimizerAdamMem, self).__init__(params, betas=betas, eps=eps, weight_decay=weight_decay) 
        self.threshold = threshold
        for group in self.param_groups:
            for p in group['params']: 
                p.m = torch.zeros_like(p.data).to(p.data.device)
                p.v = torch.zeros_like(p.data).to(p.data.device)
                p.mhat = torch.zeros_like(p.data).to(p.data.device)
                p.vhat = torch.zeros_like(p.data).to(p.data.device)
                p.accumulated_grad = torch.zeros_like(p.data).to(p.data.device)
                p.weight_update_count = torch.zeros_like(p.data).to(p.data.device)  
    def step(self,batch):
        total_weight_updates = 0
        for group in self.param_groups:
            for p in group['params']: 
                if p.grad is None:
                    continue
                p.m = group['betas'][0]* p.m + (1-group['betas'][0])*p.grad
                p.v = group['betas'][1]* p.v + (1-group['betas'][1])*p.grad**2

                p.mhat = p.m/(1-group['betas'][0]**batch)
                p.vhat= p.v/ (1-group['betas'][1]**batch)

                p.accumulated_grad += p.mhat / (torch.sqrt(p.vhat) + self.param_groups[0]['eps'])

                dw = -torch.fix(p.accumulated_grad / self.threshold)
                p.accumulated_grad += dw*self.threshold
                p.data = p.memristor.update_weights(dw)

                update_indicator = (torch.abs(dw) > 0).float()
                p.weight_update_count += update_indicator
                total_weight_updates += update_indicator.sum().item()
        return total_weight_updates
    
class OptimizerSGDMem(torch.optim.SGD):
    def __init__(self, params, momentum=0.0, weight_decay=0, threshold=10.0): 
        super(OptimizerSGDMem, self).__init__(params, momentum=momentum, weight_decay=weight_decay) 
        self.threshold = threshold
        for group in self.param_groups:
            for p in group['params']: 
                p.b = torch.zeros_like(p.data).to(p.data.device)
                p.accumulated_grad = torch.zeros_like(p.data).to(p.data.device)
                p.weight_update_count = torch.zeros_like(p.data).to(p.data.device)

    def step(self,batch):
        total_weight_updates = 0
        for group in self.param_groups:
            for p in group['params']: 
                if p.grad is None:
                    continue

                p.b = group['momentum'] * p.b + (1 - group['momentum']) * p.grad
                p.accumulated_grad += p.b

                dw = -torch.fix(p.accumulated_grad / self.threshold)
                p.accumulated_grad += dw*self.threshold

                p.data = p.memristor.update_weights(dw)

                update_indicator = (torch.abs(dw) > 0).float()
                p.weight_update_count += update_indicator
                total_weight_updates += update_indicator.sum().item()
        return total_weight_updates
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    if not os.path.isdir('memristor_ltp_ltd'):
        os.mkdir('memristor_ltp_ltd')

    
    memristor_parameters = {
        'mode': 'lin',
        'Wmin': -1.0,
        'Wmax': 1.0,
        'Gmin': 1e-6,
        'Gmax': 1e-5,
        'alpha_p': 4, #Norlinear of LTP
        'alpha_d': 4,
        'Np': 20, #Number of pulse, HRS->LRS
        'Nd': 20, 
        'gamma_p': 1.0, #Noise of pulse
        'gamma_d': 1.0,
        'gamma_gmin': 0.05, #D2D
        'gamma_gmax': 0.05
    }


    ltp_ltd_plot(memristor_parameters,filename = 'memristor_ltp_ltd/memristor')

    memristor_parameters['mode'] = 'log'
    ltp_ltd_plot(memristor_parameters,filename = 'memristor_ltp_ltd/memristor_log')
