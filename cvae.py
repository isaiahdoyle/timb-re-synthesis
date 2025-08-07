"""Conditional Variational Autoencoder (CVAE) by Vadim Borisov

GitHub: https://github.com/unnir/cVAE
License: MIT
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from dataset import TimbreDataset
from freevc import FreeVC

torch.set_default_dtype(torch.float32)

# cuda setup
if torch.backends.mps.is_available() and False:
    print('Using device==mps')
    device = torch.device('mps')
elif torch.cuda.is_available():
    print('Using device==cuda')
    device = torch.device('cuda')
else:
    print('Using device==cpu')
    device = torch.device('cpu')

kwargs = {'num_workers': 0, 'pin_memory': True} 

# hyper params
batch_size = 4
latent_size = 20
epochs = 5


timbre_model = FreeVC(use_spk=True)

train_loader = torch.utils.data.DataLoader(
    TimbreDataset(path='audio_ratings.xlsx', root_dir='dataset/converted/',
                  timbre_model=timbre_model, train=True),
    batch_size=batch_size, shuffle=True, **kwargs
)

test_loader = torch.utils.data.DataLoader(
    TimbreDataset(path='audio_ratings.xlsx', root_dir='dataset/converted/',
                  timbre_model=timbre_model, train=False),
    batch_size=batch_size, shuffle=False, **kwargs
)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        """Initialize the conditional variational autoencoder.
        
        Args:
            feature_size: The number of features in the data to encode.
            latent_size: The number of latent variables to reduce the data to.
            class_size: The number of parameters defined in condition vectors.
        """
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 128)
        self.fc21 = nn.Linear(128, latent_size)
        self.fc22 = nn.Linear(128, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 128)
        self.fc4 = nn.Linear(128, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        """
        x: (bs, feature_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        """
        z: (bs, latent_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, cx, cz=None):
        """
        cx: condition for encoder
        cz: condition for decoder (default: cx)
        """
        mu, logvar = self.encode(x.view(-1, 256), cx)
        z = self.reparameterize(mu, logvar)
        cz = cx if cz is None else cz
        return self.decode(z, cz), mu, logvar

# create a CVAE model
model = CVAE(256, latent_size, 4).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 256), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        data = batch['timbre']
        labels = batch['labels']
        data, labels = data.to(device), labels.to(device)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data = batch['timbre']
            labels = batch['labels']
            data, labels = data.to(device), labels.to(device)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            # if i == 0:
            #     n = min(data.size(0), 5)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(-1, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'reconstruction_' + str(f"{epoch:02}") + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

'''TRAINING EXAMPLE:'''

for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            c = torch.tensor([[3, 3, 3, 3]])
            sample = torch.randn(1, latent_size).to(device)
            sample = model.decode(sample, c).cpu()

torch.save(model.state_dict(), 'cvae.pth')

''''''

'''INFERENCE EXAMPLE:

model.load_state_dict(torch.load('cvae.pth'))
model.eval()  # for inference

content = timbre_model.get_content('input/audio.wav')
timbre = timbre_model.get_timbre('input/audio.wav')

# c = [[breathiness, pitch, smoothness, tone]]
modified_timbre = model(timbre, torch.tensor([[3, 3, 3, 3]]))[0]

timbre_model.synthesize(content, modified_timbre, 'output/audio-cvae.wav')

'''
