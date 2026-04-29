"""Monitor de temperatura da GPU com auto-desligamento.

Este script monitora a temperatura da GPU durante o treinamento
e desliga o PC automaticamente se a temperatura exceder 85°C.

Uso:
    from scripts.gpu_monitor import GPUMonitor
    monitor = GPUMonitor(max_temp=85, check_interval=60)
    monitor.check_temperature()  # Verifica e desliga se necessário
"""
import os
import sys
import platform
import subprocess
import time
import re
from pathlib import Path

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlGetDeviceCount,
        nvmlGetHandleByIndex, nvmlDeviceGetTemperature,
        nvmlDeviceGetName, NVML_TEMPERATURE_GPU
    )
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """Monitor de temperatura da GPU."""
    
    def __init__(self, max_temp=85, warning_temp=70, check_interval=60, gpu_index=0):
        """
        Args:
            max_temp: Temperatura máxima (°C) — acima disso, desliga o PC
            warning_temp: Temperatura de aviso (°C) — mostrar alerta (padrão: 70°C)
            check_interval: Intervalo de verificação em segundos (padrão: 60s)
            gpu_index: Índice da GPU a monitorar (padrão: 0)
        """
        self.max_temp = max_temp
        self.warning_temp = warning_temp
        self.check_interval = check_interval
        self.gpu_index = gpu_index
        self.enabled = False
        self.use_pynvml = False
        self.use_nvidia_smi = False
        self.last_check_time = 0
        
        # Tentar inicializar com pynvml
        if PYNVML_AVAILABLE:
            try:
                nvmlInit()
                device_count = nvmlGetDeviceCount()
                if gpu_index >= device_count:
                    print(f"❌ GPU index {gpu_index} não existe (encontrados {device_count} GPUs)")
                else:
                    self.handle = nvmlGetHandleByIndex(gpu_index)
                    gpu_name = nvmlDeviceGetName(self.handle).decode('utf-8')
                    print(f"✓ GPU Monitor ativado via pynvml: {gpu_name} (index {gpu_index})")
                    self.enabled = True
                    self.use_pynvml = True
            except Exception as e:
                print(f"⚠️  pynvml falhou ({e}). Tentando nvidia-smi...")
        
        # Fallback: usar nvidia-smi
        if not self.enabled:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_name = result.stdout.strip()
                    print(f"✓ GPU Monitor ativado via nvidia-smi: {gpu_name} (index {gpu_index})")
                    self.enabled = True
                    self.use_nvidia_smi = True
            except Exception as e:
                print(f"❌ nvidia-smi também falhou: {e}")
                print("⚠️  Monitoramento de temperatura desabilitado")
    
    def get_temperature(self):
        """Retorna a temperatura atual da GPU em °C, ou None se não disponível."""
        if not self.enabled:
            return None
        
        try:
            if self.use_pynvml:
                return nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
            elif self.use_nvidia_smi:
                result = subprocess.run(
                    ["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    temp_str = result.stdout.strip()
                    # nvidia-smi retorna "XX C" format
                    temp_match = re.search(r'(\d+)', temp_str)
                    if temp_match:
                        return int(temp_match.group(1))
        except Exception as e:
            print(f"⚠️  Erro ao ler temperatura: {e}")
        
        return None
    
    def check_temperature(self):
        """Verifica temperatura e desliga o PC se necessário. Retorna True se OK, False se desligou."""
        if not self.enabled:
            return True
        
        # Respeita o intervalo de verificação
        now = time.time()
        if now - self.last_check_time < self.check_interval:
            return True
        self.last_check_time = now
        
        temp = self.get_temperature()
        if temp is None:
            return True
        
        # Aviso de temperatura alta
        if temp >= self.warning_temp:
            print(f"⚠️  GPU QUENTE: {temp}°C >= {self.warning_temp}°C (limite: {self.max_temp}°C)")
        
        # Desligamento automático
        if temp >= self.max_temp:
            print(f"\n{'=' * 60}")
            print(f"🚨 ALERTA CRÍTICO: GPU em {temp}°C >= {self.max_temp}°C")
            print(f"🔴 DESLIGANDO O PC EM 30 SEGUNDOS...")
            print(f"{'=' * 60}\n")
            
            try:
                time.sleep(5)  # Dar tempo para salvar buffers
                self._shutdown_pc()
                return False
            except Exception as e:
                print(f"❌ Erro ao desligar: {e}")
                return False
        
        return True
    
    def _shutdown_pc(self):
        """Desliga o PC via comando do SO."""
        if platform.system() == "Windows":
            # Windows: shutdown /s /t 60 = desligamento em 60 segundos
            subprocess.run(["shutdown", "/s", "/t", "30"], check=False)
        elif platform.system() in ["Linux", "Darwin"]:
            # Linux/Mac: shutdown -h +30 = desligamento em 30 minutos (aprox)
            subprocess.run(["sudo", "shutdown", "-h", "+0.5"], check=False)
        else:
            print("❌ Sistema operacional não suportado para desligamento automático")
    
    def cleanup(self):
        """Limpa recursos da NVIDIA."""
        if self.use_pynvml and PYNVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass


def monitor_temperature_in_thread(monitor, stop_event=None):
    """Executa monitoramento em loop (para uso em thread)."""
    while stop_event is None or not stop_event.is_set():
        if not monitor.check_temperature():
            break
        time.sleep(monitor.check_interval)


if __name__ == "__main__":
    # Teste
    monitor = GPUMonitor(max_temp=85, warning_temp=70)
    print(f"Temperatura atual: {monitor.get_temperature()}°C")
    monitor.cleanup()
