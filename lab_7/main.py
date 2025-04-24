# 1. создание случайных бит на передачу длиной N
# 2. Формирование квадратурных BPSK и ASK
# 3. Добавление белого комплексного шума с расчетом в db 
# 4. демодуляция из bpsk и ask
# 5. сравнение принятых бит и изначальных для bpsk и ask
# 6. расчет BER для этих сообщений

# использовать не жестике решения для принятия решения для демодуляции символа в бит, а зачатки мягкого с помощью сравнения расстояния принятой точки до референсных
# вывести график созвездий для bpsk и ask 

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def generate_random_bits(N):
    return np.random.randint(0, 2, N)

def modulate_bpsk(bits):
    # BPSK: 0 -> -1, 1 -> 1
    return 2 * bits - 1

def modulate_ask(bits):
    # ASK: 0 -> 0, 1 -> 1
    return bits.astype(float)

def add_noise(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def demodulate_soft(signal, reference_symbols):
    distances = np.abs(signal[:, np.newaxis] - reference_symbols)
    decisions = np.argmin(distances, axis=1)
    return decisions

def calculate_ber(original_bits, received_bits):
    errors = np.sum(original_bits != received_bits)
    return errors / len(original_bits)

if __name__ == "__main__":
    N = 50000  
    snr_db_values = np.arange(-10, 12, 2) 

    ber_bpsk = []
    ber_ask = []
    ber_ask_theory = []
    ber_psk_theory = []

    original_bits = generate_random_bits(N)

    bpsk_reference_symbols = np.array([-1, 1])  # BPSK: -1 и 1
    ask_reference_symbols = np.array([0, 1])   # ASK: 0 и 1

    for snr_db in snr_db_values:
        snr = 10 ** (snr_db / 10)
        # Для ASK
        ber_ask_theory.append(q_function(np.sqrt(2 * snr)))
        # Для BPSK
        ber_psk_theory.append(q_function(np.sqrt(snr)))

    for snr_db in snr_db_values:
        # BPSK 
        bpsk_signal = modulate_bpsk(original_bits)
        noisy_bpsk_signal = add_noise(bpsk_signal, snr_db)
        received_bpsk_bits = demodulate_soft(np.real(noisy_bpsk_signal), bpsk_reference_symbols)

        # ASK 
        ask_signal = modulate_ask(original_bits)
        noisy_ask_signal = add_noise(ask_signal, snr_db)
        received_ask_bits = demodulate_soft(np.real(noisy_ask_signal), ask_reference_symbols)

        # BER
        ber_bpsk.append(calculate_ber(original_bits, received_bpsk_bits))
        ber_ask.append(calculate_ber(original_bits, received_ask_bits))

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_values, ber_bpsk, label="BPSK BER", marker="o")
    plt.semilogy(snr_db_values, ber_ask, label="ASK BER", marker="s")
    plt.semilogy(snr_db_values, ber_ask_theory, label="ASK THEORY", marker="s")
    plt.semilogy(snr_db_values, ber_psk_theory, label="BPSK THEORY", marker="s")
    plt.title("BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()