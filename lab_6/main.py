import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

fs = 100
ts = 1 / fs
Ts = ts  # Длительность символа (один отсчет)
N_symbols = 10000
A = np.sqrt(2)
snr_db_range = np.arange(0, 16, 1)  # Диапазон SNR от 0 до 15 дБ

ber_ask_theory = []
ber_psk_theory = []
ber_ask_exp = []
ber_psk_exp = []

# Теоретические расчеты BER
for snr_db in snr_db_range:
    snr = 10 ** (snr_db / 10)
    # Для ASK
    ber_ask_theory.append(q_function(np.sqrt(2 * snr)))
    # Для BPSK
    ber_psk_theory.append(q_function(np.sqrt(snr)))

# Экспериментальные расчеты BER
for snr_db in snr_db_range:
    snr_linear = 10 ** (snr_db / 10)
    
    # Генерация битовых последовательностей
    bits_ask = np.random.randint(0, 2, N_symbols)
    bits_bpsk = np.random.randint(0, 2, N_symbols)
    
    symbols_ask = A * (2 * bits_ask - 1)
    symbols_bpsk = (2 * bits_bpsk - 1)
    
    E_ask = (A ** 2) * Ts 
    E_bpsk = (1 ** 2) * Ts 
    
    sigma_ask = np.sqrt(E_ask / snr_linear)
    sigma_bpsk = np.sqrt(E_bpsk / snr_linear)
    
    noise_ask = np.random.normal(0, sigma_ask* 7.5, N_symbols)
    noise_bpsk = np.random.normal(0, sigma_bpsk* 10, N_symbols)
    print(N_symbols)
    print(sigma_ask)
    print(noise_ask)
    
    # Сигналы с шумом
    signal_ask = symbols_ask + noise_ask
    signal_bpsk = symbols_bpsk + noise_bpsk
    
    # Демодуляция ASK с порогом Vt = sqrt(E_ask)/2
    Vt = np.sqrt(E_ask) / 2
    decision_ask = (signal_ask > Vt).astype(int)
    errors_ask = np.sum(decision_ask != bits_ask)
    ber_ask_exp.append(errors_ask / N_symbols)
    
    # Демодуляция BPSK (сравнение с порогом 0)
    decision_bpsk = (signal_bpsk > 0).astype(int)
    errors_bpsk = np.sum(decision_bpsk != bits_bpsk)
    ber_psk_exp.append(errors_bpsk / N_symbols)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.semilogy(snr_db_range, ber_ask_theory, 'r^-', label='ASK Theory', alpha=0.7)
plt.semilogy(snr_db_range, ber_psk_theory, 'bs-', label='BPSK Theory', alpha=0.7)
plt.semilogy(snr_db_range, ber_ask_exp, 'go--', label='ASK Experiment', alpha=0.7)
plt.semilogy(snr_db_range, ber_psk_exp, 'kD--', label='BPSK Experiment', alpha=0.7)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs SNR for ASK and BPSK')
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()