from diffusers import UNet2DModel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
from inception import fid_inception_v3

class LargeConvDenoiserNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: list[int] = [64, 128, 256, 512, 1024],
        layers_per_block: int = 2,
        downblock: str = 'ResnetDownsampleBlock2D',
        upblock: str = 'ResnetUpsampleBlock2D',
        add_attention: bool = True,
        attention_head_dim: int = 64,
        low_condition: bool = False,
        timestep_condition: bool = True,
        global_skip_connection: bool = True,
        num_class_embeds: int | None = None,
    ):
        super().__init__()
        self.low_condition = low_condition
        self.timestep_condition = timestep_condition
        self.global_skip_connection = global_skip_connection
        self.divide_factor = 2 ** len(channels)

        in_channels = 2 * in_channels if self.low_condition else in_channels

        self.backbone = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=channels,
            layers_per_block=layers_per_block,
            down_block_types=tuple(downblock for _ in range(len(channels))),
            up_block_types=tuple(upblock for _ in range(len(channels))),
            add_attention=add_attention,
            attention_head_dim=attention_head_dim,
            num_class_embeds=num_class_embeds,
        )

    def padding(self, x):
        _, _, W, H = x.shape
        desired_width = (
            (W + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor
        desired_height = (
            (H + self.divide_factor - 1) // self.divide_factor
        ) * self.divide_factor

        # Calculate the padding needed
        padding_w = desired_width - W
        padding_h = desired_height - H

        return F.pad(x, (0, padding_h, 0, padding_w), mode="constant", value=0), W, H

    def remove_padding(self, x, W, H):
        return x[:, :, :W, :H]

    def forward(self, x_t, t):

        # add padding to fit nearest value divisible by self.divide_factor
        x_in, W, H = self.padding(x_t)

        model_output = self.backbone(
            x_in,
            timestep=t if self.timestep_condition else 0,
        ).sample

        model_output = self.remove_padding(model_output, W, H)

        if self.global_skip_connection:
            model_output[:, :3] = model_output[:, :3] + x_t

        return model_output  # pred_x_0
    
def apply(
        coefficients: np.array, 
        timesteps: torch.tensor, 
        x: torch.tensor
        ):
    """
    Wyznacza współczynniki z aktualnych kroków dyfuzyjnych i przemnaża je przez tensor x.

    :param coefficients: 1D numpy array, wartości współczynników noise scheduler'a (np alpha, beta itp) dla wszystkich 1000 kroków
    :param timesteps: 1D tensor, wartości kroków dyfuzjnych dla każdej z próbki w batchu
    :param x: tensor, batch wartości z domeny danych (czystych bądź zaszumionych)
    :return: tensor, wynik przemnożenia coefficients[t] * x. Wynik posiada taki sam rozmiar jak x.
    """
    factors = torch.from_numpy(coefficients).to(device=timesteps.device)[timesteps].float() 
    K = x.dim() - 1 # ilość osi w x (minus 1 bo nie liczymy batch size), dla danych Moons jest to 1, dla obrazów jest to 3 (kanał, szerokość, wysokość)
    factors = factors.view(-1, *([1]*K)).expand(x.shape) # tworzymy puste osie, żeby można przemnożyć mnożniki przez x
    return factors * x


class GaussianDiffusion:
    """
    Podstawowa klasa architektury DDPM. 
    """
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.timesteps = list(range(self.num_timesteps))[::-1]

        self.betas = np.linspace(start=0.0001, stop=0.02, num=self.num_timesteps)
        self.setup_noise_scheduler(self.betas)

    def setup_noise_scheduler(self, betas):
        self.alphas = 1.0 - betas
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
        self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])

    def q_sample(self, x_0, t, noise=None):
        """
        próbkowanie z rozkładu q(x_t | x_0). dodaje szum do próbki x_0 o sile zgodnie ze znacznikiem czasowym t.

        :x_0: batch czystych próbek do zaszumienia
        :t: wektor znaczników czasowych. Każda próbka ma swój niezależny czas.
        :noise: Szum nakładany na próbkę x_0, przeskalowany zgodnie z wartościami schedulera. 
                Gdy szum nie został podany, jest on próbkowany z rozkładu Gaussa
        :return: x_t - batch zaszumionych próbek x_0 do chwili czasowej t
        """
        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)

        # POCZĄTEK ROZWIĄZANIA
        return(apply(np.sqrt(self.alpha_bars), t, x_0) + apply(np.sqrt(1 - self.alpha_bars), t, noise))
        # KONIEC ROZWIĄZANIA

    def q_posterior(self, x_t, x_0, t, noise=None):
        """
        próbkowanie z rozkładu q(x_{t-1} | x_t, x_0). Usuwa część szumu, modelując jak powinna wyglądać próbka x_{t-1}, by x_t była prawdopodobna.

        :x_t: batch zaszumionych próbek w chwili t
        :x_0: batch czystych próbek 
        :t: wektor znaczników czasowych. Każda próbka ma swój niezależny czas.
        :noise: szum służący do wypróbkowania konkretnego x_{t-1} z rozkładu a posteriori, które ma rozkład Gaussa.
        :return: x_{t-1} - batch delikatnie odszumionych próbek w chwili czasowej t-1. 
        """
        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)

        # POCZĄTEK ROZWIĄZANIA
        alpha_t = apply(self.alphas, t, torch.ones_like(x_t).to(x_t.device))
        alpha_bar_t = apply(self.alpha_bars, t, torch.ones_like(x_t).to(x_t.device))
        alpha_bar_prev_t = apply(self.alpha_bars_prev, t, torch.ones_like(x_t).to(x_t.device))
        beta_t = apply(self.betas, t, torch.ones_like(x_t))
        
        coef1 = ((torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t)) / (1.0 - alpha_bar_t))
        coef2 = ((torch.sqrt(alpha_bar_prev_t) * beta_t) / (1.0 - alpha_bar_t))
        mean = coef1 * x_t + coef2 * x_0

        var = ((1.0 - alpha_bar_prev_t) * beta_t) / (1.0 - alpha_bar_t)
        std = torch.sqrt(var)

        return mean + std * noise

        # KONIEC ROZWIĄZANIA

    def _predict_x_0_from_eps(self, x_t, t, eps):
        """
        Estymuje x_0 na bazie zaszumionych próbek x_t, szumu jaki się w nich znajduje eps, oraz znacznika czasowego t. 
        Można łatwo wyznaczyć ten wzór przekształcając wzór (4) (stosując sztuczkę z reparametryzacją).

        :x_t: zaszumione próbki x_t
        :t: znaczniki czasowe. Różne dla innych próbek w batchu
        :eps: szum obecny w x_t
        :return: próbki x_0 stworzone poprzez usunięcie szumu eps z próbek x_t.
        """
        return (
            apply(np.sqrt(1.0 / self.alpha_bars), t, x_t) - 
            apply(np.sqrt(1.0 / self.alpha_bars - 1.0), t, eps)
        )

    def _predict_eps_from_xstart(self, x_t, t, x_0):
        """
        Estymuje szum w próbce x_t na bazie zaszumionych próbek x_t, czystych zdjęć x_0, oraz znacznika czasowego t. 
        Można łatwo wyznaczyć ten wzór przekształcając wzór (4) (stosując sztuczkę z reparametryzacją).

        :x_t: zaszumione próbki x_t
        :t: znaczniki czasowe. Różne dla innych próbek w batchu
        :x_0: Czyste próbki, bądź estymaty czystych próbek.
        :return: próbki x_0 stworzone poprzez usunięcie szumu eps z próbek x_t.
        """
        return (
            apply(np.sqrt(1.0 / self.alpha_bars), t, x_t) - x_0
        ) / apply(np.sqrt(1.0 / self.alpha_bars - 1.0), t, torch.ones_like(x_t))

    def train_losses(self, model, x_0, DEVICE):
        """
        Funkcja która zwraca wartość funkcji straty z wzoru (11) dla pewnego batcha czystych próbek x_0 oraz modelu.
        Funkcja próbkuje losowe znaczniki czasowe, oraz szum aplikowany do danych x_0. 
        Następnie za pomocą q_sample tworzy zaszumione dane x_t zgodnie z definicją procesu w przód.
        Model predykuje wartość szumu na podstawie danych x_t oraz znacznika t. 
        Ostateczny wynik to MSE pomiędzy prawdziwym (wcześniej wypróbkowanym) szumem, a tym wypredykowanym przez model

        :model: odszumiająca sieć neuronowa, której rolą jest predykować szum 
        :x_0: Prawdziwe dane, na których uczymy model. 
        :return: wartość błedu średniokwadratowego pomiędzy prawdziwym a predykowanym szumem Gaussowskim
        """
        # POCZĄTEK ROZWIĄZANIA
        batch_size = x_0.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size, ), device=DEVICE)
        eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps)
        eps_pred = model(x_t, t)
        loss = torch.nn.functional.mse_loss(eps_pred, eps)
        # KONIEC ROZWIĄZANIA

        return loss

    @torch.no_grad()
    def p_sample_loop(
        self, 
        model: nn.Module, 
        noise: torch.tensor, 
        num_inference_steps: int = 1000,
        return_trajectory: bool = False, 
        clip: bool = False,
        quiet: bool = True,
        device: str = "cpu"
        ):
        """
        Proces dyfuzyjny wstecz służący do generowania nowych próbek. 
        Proces ten zaczyna się na czystym szumie, który jest wielokrotnie poddawany stopniowemu
        odszumianiu z wykorzystaniem sieci neuronowej. 

        :model: sieć neuronowa predykująca szum. Na podstawie jej wyników dokonywane jest generowanie danych
        :noise: szum startowy. Wygenerowanie dane będą posiadały dokładnie ten sam wymiar
        :num_inference_steps: Ilość kroków dyfuzyjnych w procesie próbkowania. Im więcej tym lepsze wyniki modelu, ale wolniejsza inferencja
        :return_trajectory: czy zwracać wyniki kroków pośrednich x_t dla wszystkich 1000 kroków (dla celów wizualizacyjnych)
        :clip: Czy aplikować clip do przedziału danych na estymaty x_0 w każdym kroku. Szczególnie użyteczne dla zdjęć.
        :quiet: czy nie wyświetlać paska postępu

        :return: ostateczne próbki jako macierz numpy. Gdy return_trajectory jest True - dodatkowo sekwencje x_t
        """
        self._respace(num_timesteps=num_inference_steps)

        x_t = noise.to(device)
        bsz = x_t.shape[0]
        trajectory = [x_t.clone().cpu()]

        # iterujemy zaczynając od T i zmniejszając kroki aż do 0
        pbar = tqdm(enumerate(self.timesteps), desc='Próbkowanie', total=self.num_timesteps) if not quiet else enumerate(self.timesteps)

        for idx, time in pbar:
            t = torch.tensor([time] * bsz, device=x_t.device).long()
            i = torch.tensor([self.num_timesteps - idx - 1] * bsz, device=x_t.device).long()

            # próbkuj szum poprzez model i wyznacz estymatę x_0 
            eps = model(x_t, t)
            x_0 = self._predict_x_0_from_eps(x_t, i, eps)

            # W sytuacji kiedy model operuje na zdjęciach, często normalizujemy zdjęcia do przedziału (-1,1)
            # Gdy w każdym kroku clipujemy estymację czystej próbki x_0 do tego samego przedziału zwiększa to stabilność modelu
            if clip:
                x_0 = x_0.clamp(-1, 1)

            # krok odszumiający przy wykorzystaniu wzoru na posterior procesu w przód
            x_t = self.q_posterior(x_t, x_0, i)

            # dodanie x_t do celów wizualizacyjnych (nie ma wpływu na działanie metody)
            trajectory.append(x_t.clone().cpu())

        self._respace(1000) # na koniec wracamy do domyślnych 1000 kroków

        if return_trajectory:
            return x_0.cpu().numpy(), torch.stack(trajectory, dim=0).numpy()
        return x_0.cpu().numpy()
    
    def _respace(self, num_timesteps):
        """
        Funkcja zmieniająca ilość kroków dyfuzyjnych w inferencji. 
        Redukcja kroków wiąże się z szybszą inferencją kosztem gorszej jakości.
        Wartości noise scheduler'a muszą zostać dopasowane, ponieważ różnice pomiędzy x_{t-1} a x_t ulegają zmianie

        :num_timesteps: nowa ilość kroków dyfuzyjnych.
        """
        betas = np.linspace(start=0.0001, stop=0.02, num=1000)
        self.setup_noise_scheduler(betas) 

        self.num_timesteps = num_timesteps
        self.timesteps = np.linspace(999, 0, self.num_timesteps, dtype=int, endpoint=True)

        last_alpha_cumprod = 1.0

        self.betas = []

        for i, alpha_bar in enumerate(self.alpha_bars):
            if i in self.timesteps:
                self.betas.append(1 - alpha_bar / last_alpha_cumprod)
                last_alpha_cumprod = alpha_bar
        
        self.betas = np.array(self.betas)
        self.setup_noise_scheduler(self.betas)

class DeterministicGaussianDiffusion(GaussianDiffusion):
    def q_posterior(self, x_t, x_0, t):
        """
        próbkowanie z rozkładu q(x_{t-1} | x_t, x_0) korzystając z metody DDIM.
        Usuwa część szumu, interpolując pomiędzy x_t a x_0

        :x_t: batch zaszumionych próbek w chwili t
        :x_0: batch czystych próbek 
        :t: wektor znaczników czasowych. Każda próbka ma swój niezależny czas.
        :noise: szum służący do wypróbkowania konkretnego x_{t-1} z rozkładu a posteriori, które ma rozkład Gaussa.
        :return: x_{t-1} - batch delikatnie odszumionych próbek w chwili czasowej t-1. 
        """

        # POCZĄTEK ROZWIĄZANIA
        eps = self._predict_eps_from_xstart(x_t, t, x_0)

        alpha_bar_prev_t = apply(self.alpha_bars_prev, t, torch.ones_like(x_t).to(x_t.device))

        x_prev = torch.sqrt(alpha_bar_prev_t) * x_0 + torch.sqrt(1 - alpha_bar_prev_t) * eps

        return x_prev
        # KONIEC ROZWIĄZANIA

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3,  # Final average pooling features
    }

    def __init__(
        self,
        output_blocks=(DEFAULT_BLOCK_INDEX,),
        resize_input=True,
        normalize_input=True,
        requires_grad=False,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, "Last possible output block index is 3"

        self.blocks = nn.ModuleList()

        inception = fid_inception_v3()

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x.squeeze(-1).squeeze(-1).cpu())

            if idx == self.last_needed_block:
                break

        if len(self.output_blocks) == 1:
            return outp[0]

        return outp
