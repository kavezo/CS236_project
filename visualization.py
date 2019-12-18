from codebase import utils as ut
from codebase.models import conv_vae as model

import numpy as np
import matplotlib.pyplot as plt
import torch

model_name = 'model=convvae_z=60_run=0000'
device='cpu'

vae = model.ConvVAE(z_dim=60, name=model_name, k=6).to(device)
ut.load_model_by_name(vae, global_step=5000, device=device)

path = 'data/hiyori/mel/'
tests = ['- あ', '- い', '- か', '- き', 'a な', 'i に']

zs = []

for name in tests:
    x = np.load(path+name+'.npy').real.astype('float32')
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    x = torch.from_numpy(x[:,:157].reshape(1,1,80,157))
    phi = vae.enc.encode(x)
    zs.append(phi[0].detach().numpy())
    z_hat = ut.sample_gaussian(*phi)
    x_hat = vae.sample_x_given(z_hat).detach().numpy().reshape(80, 157)

    # fig, axs = plt.subplots(2)
    # axs[0].imshow(x_hat)
    # axs[1].imshow(x.reshape(80, 157))
    # plt.show()

print('a-i: ' + str(np.linalg.norm(zs[0]-zs[1])))
print('a-ka: ' + str(np.linalg.norm(zs[0]-zs[2])))
print('(ka+(i-a))-ki: ' + str(np.linalg.norm(zs[2]+(zs[1]-zs[0])-zs[3])))
print('(na+(i-a))-ni: ' + str(np.linalg.norm(zs[4]+(zs[1]-zs[0])-zs[5])))

# x = vae.sample_x(10).squeeze().detach().numpy()
# rows = []
# for i in range(20):
#     rows.append(np.concatenate(x[i*10:(i+1)*10,:,:], 1))
# print(rows[0].shape)
#full = np.concatenate(rows)
# for i in range(10):
#     plt.imshow(x[i])
#     plt.show()
#plt.imsave('output/convvae.png', full)