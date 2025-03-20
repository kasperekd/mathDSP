import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

Fs = 48000  # Частота дискретизации
N = 1024    # Длина сигнала

np.random.seed(1)
white_noise = np.random.randn(N)

order_low = 6
rp_low = 0.5 
rs_low = 50  
wp_low = 3000 # Частота среза
wn_low = wp_low / (Fs / 2)  # Нормированная частота

b_low, a_low = sig.ellip(order_low, rp_low, rs_low, wn_low, btype='low', analog=False)
filtered_low = sig.lfilter(b_low, a_low, white_noise)

# ПФ (эллиптический фильтр)
order_band = 6
rp_band = 0.5
rs_band = 50
wp_band = [3000, 6000]
wn_band = [w / (Fs / 2) for w in wp_band]

b_band, a_band = sig.ellip(order_band, rp_band, rs_band, wn_band, btype='band', analog=False)
filtered_band = sig.lfilter(b_band, a_band, white_noise)

# Функция вычисления АКФ
def autocorr(x):
    return np.correlate(x - np.mean(x), x - np.mean(x), mode='full')[len(x)-1:]

# Вычисление АКФ
acf_white = autocorr(white_noise)
acf_low = autocorr(filtered_low)
acf_band = autocorr(filtered_band)

# Вычисление СПМ 
f_white, Pxx_white = sig.welch(white_noise, Fs, nperseg=N//2)
f_low, Pxx_low = sig.welch(filtered_low, Fs, nperseg=N//2)
f_band, Pxx_band = sig.welch(filtered_band, Fs, nperseg=N//2)

# Усреднение 1000 реализаций для ФНЧ
n_realizations = 1000
nperseg = N//2
spm_avg_low = np.zeros(nperseg//2 + 1)  
acf_avg_low = np.zeros(N)

for _ in range(n_realizations):
    sig_realization = np.random.randn(N)
    filtered = sig.lfilter(b_low, a_low, sig_realization)
    acf_avg_low += autocorr(filtered)
    
    f, Pxx = sig.welch(filtered, Fs, nperseg=nperseg)
    spm_avg_low += Pxx

acf_avg_low /= n_realizations
spm_avg_low /= n_realizations

# Интервал корреляции для ФНЧ
acf_max = acf_avg_low[0]
threshold = acf_max / np.exp(1)
idx = np.where(acf_avg_low < threshold)[0][0]
correlation_interval = idx * (1/Fs)
print(f"Интервал корреляции: {correlation_interval:.2f} секунд")

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(acf_white[:100], label='Белый шум')
plt.plot(acf_low[:100], label='ФНЧ')
plt.plot(acf_band[:100], label='ПФ')
plt.title('АКФ')
plt.xlabel('Отсчет')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(f_white, Pxx_white, label='Белый шум')
plt.plot(f_low, Pxx_low, label='ФНЧ')
plt.plot(f_band, Pxx_band, label='ПФ')
plt.title('СПМ')
plt.xlabel('Частота, Гц')
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(acf_avg_low[:100], label='Усредненная АКФ ФНЧ')
plt.title('АКФ после усреднения')
plt.xlabel('Отсчет')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(f_white[:len(spm_avg_low)], spm_avg_low, label='Усредненная СПМ ФНЧ')
plt.title('СПМ после усреднения')
plt.xlabel('Частота, Гц')
plt.grid()

plt.tight_layout()
plt.show()

Fs = 48000  # Частота дискретизации
N = 1024    # Длина сигнала

b_up = 1
a_up = [1, -0.9]  

np.random.seed(1) 
sig_in = np.random.randn(N)
sig_up = sig.lfilter(b_up, a_up, sig_in)

chl = np.array([
    0.112873323983817 + 0.707197839105975j,
    0.447066470535225 - 0.375487664593309j,
    0.189507622364489 - 0.132430466488543j,
    0.111178811342494 - 0.0857241111225552j
])

h = np.array([1,0.7,0.5,0.3])
# y = np.convolve(sig_up, chl, mode='full')
y2 = np.convolve(sig_in, h, mode='same')


def crosscorr(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.correlate(x, y, mode='full')

ccf = crosscorr(y2, sig_in )
# ccf = crosscorr(sig_up, y)
print(ccf[0:10])
def autocorr(x):
    return np.correlate(x - np.mean(x), x - np.mean(x), mode='full')[len(x)-1:]

acf_up = autocorr(sig_up)
# print(acf_up)
f_up, Pxx_up = sig.welch(sig_up, Fs, nperseg=N//2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(acf_up)
plt.title('АКФ УП СП')
plt.xlabel('Отсчет')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(f_up, Pxx_up)
plt.title('СПМ УП СП')
plt.xlabel('Частота, Гц')
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(ccf[0:10])
plt.title('ККФ между УП СП и выходом через chl')
plt.xlabel('Отсчет')
plt.grid()

plt.tight_layout()
plt.show()