# Нужно:
# 1. Создать биты на передачу {0,1} длины N
# 2. Применить модуляцию (1 -> 1; 0 -> -1)
# 3. Задать параметр oversampling (количество семплов для модуляции одного символа)
# 4. Создать массив длины количества сэмплов белого шума с возможностью задавать параметры
# 5. Сложить символы с оверсемплингом с шумом
# 6. Подать сигнал с шумом на коррелятор (корреляция с опорным сигналом (допустим 1))
# 7. Сравнение его с порогом. Если > 0, то принятый сигнал 1 иначе -1
# 8. Перевод символов в биты
# 9. Вывод BER SER

# power = L*(1/N) SUM^N_i=1 (|S_1|^2)

import numpy as np
import matplotlib.pyplot as plt

def generate_bits(N):
    return np.random.randint(0, 2, N)

def modulate(bits):
    return 2 * bits - 1

def oversample(signal, oversampling_factor):
    return np.repeat(signal, oversampling_factor)

def calculate_signal_power(signal, oversampling_factor):
    # P_s = (L / N) * SUM(|S_i|^2)
    N = len(signal) // oversampling_factor 
    sum_ =  np.sum(np.abs(signal)**2)
    power = (oversampling_factor / N) * sum_
    # print(sum_)
    # power = 1
    return power

def generate_noise(length, signal_power, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    return np.random.normal(0, noise_std, length)

def add_noise(signal, noise):
    return signal + noise

def correlate(signal, oversampling_factor):
    reference_signal = np.ones(oversampling_factor)
    correlated_signal = np.convolve(signal, reference_signal, mode='valid')
    return correlated_signal[::oversampling_factor]

def threshold_decision(correlated_signal):
    return np.where(correlated_signal > 0, 1, -1)

def demodulate(symbols):
    return (symbols + 1) // 2

def calculate_errors(original_bits, received_bits):
    bit_errors = np.sum(original_bits != received_bits)
    symbol_errors = np.sum((original_bits * 2 - 1) != received_bits)
    ber = bit_errors / len(original_bits)
    # ser = symbol_errors / len(original_bits)
    return ber

def simulate_ber_vs_snr(N, oversampling_factor, snr_range):
    ber_values = []
    for snr_db in snr_range:
        bits = generate_bits(N)
        symbols = modulate(bits)
        sampled_signal = oversample(symbols, oversampling_factor)
        signal_power = calculate_signal_power(sampled_signal, oversampling_factor)
        noise = generate_noise(len(sampled_signal), signal_power, snr_db)
        noisy_signal = add_noise(sampled_signal, noise)
        correlated_signal = correlate(noisy_signal, oversampling_factor)
        received_symbols = threshold_decision(correlated_signal)
        received_bits = demodulate(received_symbols)
        ber = calculate_errors(bits, received_bits)
        ber_values.append(ber)
    return ber_values

def main():
    N = 10000
    oversampling_factor = 8
    snr_db = 100

    snr_range = np.arange(0, 21, 0.1) 
    last_snr_db = snr_range[-1]
    ber_values = simulate_ber_vs_snr(N, oversampling_factor, snr_range)

    bits = generate_bits(N)
    # print(bits)
    symbols = modulate(bits)
    sampled_signal = oversample(symbols, oversampling_factor)
    signal_power = calculate_signal_power(sampled_signal, oversampling_factor)
    print(signal_power)
    noise = generate_noise(len(sampled_signal), signal_power, snr_db)
    noisy_signal = add_noise(sampled_signal, noise)

    correlated_signal = correlate(noisy_signal, oversampling_factor)
    received_symbols = threshold_decision(correlated_signal)
    received_bits = demodulate(received_symbols)
    ber = calculate_errors(bits, received_bits)

    print(f"Original Bits: {bits[:10]}...")
    print(f"Received Bits: {received_bits[:10]}...")
    print(f"BER: {ber:.4e}")

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    # Оригинальный сигнал
    axs[0].plot(sampled_signal, label="Original Signal", color="blue")
    axs[0].set_title("Original Signal (After Oversampling)")
    axs[0].legend()
    axs[0].grid(True)

    # Сигнал с шумом
    axs[1].plot(noisy_signal, label="Noisy Signal", color="orange")
    axs[1].set_title(f"Signal with Noise (SNR={last_snr_db} dB)")
    axs[1].legend()
    axs[1].grid(True)

    # Корреляция
    axs[2].plot(correlated_signal, label="Correlation Output", color="green")
    axs[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[2].set_title("Correlation Output")
    axs[2].legend()
    axs[2].grid(True)

    # BER vs SNR
    axs[3].semilogy(snr_range, ber_values, marker='o', label="BER", color="red")
    axs[3].set_title("BER vs SNR")
    axs[3].set_xlabel("SNR (dB)")
    axs[3].set_ylabel("BER")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()